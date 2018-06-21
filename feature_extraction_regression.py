#################################
## Measuring Feature importance by slope/target correlation
##################################
# from extract_statistics import *
from sklearn.metrics import accuracy_score
import numpy as np
from scipy import stats
from copy import deepcopy
from multiprocessing import Pool, Manager, cpu_count, Queue
import pickle
from load_raw import load_raw
import os
import tqdm
from machine_learner import filter_features

def get_importances(slopes, target):
	#############################
	## Get coherence with target
	#############################
	slopes = slopes.reshape(slopes.shape[0], slopes.shape[2], slopes.shape[1])

	for lead in range(slopes.shape[0]):
		for feature in range(slopes.shape[1]):

			if (np.any(slopes[lead][feature] < 0)) and (np.any(slopes[lead][feature] > 0)):
				print("NOT")
				for trial in range(slopes.shape[2]):

					if slopes[lead][feature][trial] < 0:
						slopes[lead][feature][trial] = -1
					else:
						slopes[lead][feature][trial] = 1
			else:
				print("ALL PURE")
				mean = slopes[lead][feature].mean()
				for trial in range(slopes.shape[2]):
					if slopes[lead][feature][trial] < mean:
						slopes[lead][feature][trial] = -1
					else:
						slopes[lead][feature][trial] = 1

	importance = []
	# iterate over leads
	for z in range(slopes.shape[0]):
		acc = []
		for x in range(slopes.shape[1]):
			acc.append(accuracy_score(target[x].ravel(), slopes[z][x]))

		importance.append(acc)

	return importance


'''######################################
## attempt to multithread get_slopes function '''


def multithread_slope_extraction(X):
	pool = Pool(20)

	m = Manager()
	queue = m.Queue()  # create queue to save all the results
	tasks = []
	X = np.nan_to_num(X)

	X = X.reshape(X.shape[0], X.shape[1], X.shape[3], X.shape[2])

	n_features = X.shape[3]
	slopes = [[[[] for _ in range(X.shape[2])] for _ in range(X.shape[1])] for _ in range(X.shape[0])]

	x = list(range(n_features))
	for lead in range(X.shape[0]):
		for trial in range(X.shape[1]):
			for feature in range(X.shape[2]):
				f = X[lead][trial][feature].ravel()
				tasks.append([f, x, queue, lead, trial, feature])
	print(len(tasks))
	pool.map(execute_slope, tasks)  # create the results
	for _ in tqdm.tqdm(pool.imap_unordered(execute_slope, tasks), total=len(tasks)):
		pass
	while not queue.empty():
		slope, lead, trial, feature = queue.get()
		slopes[lead][trial][feature] = slope
	slopes = np.array(slopes)
	pool.close()
	pool.join()
	return slopes


def multithread_average_slope(X, removal=False, bottom_thresh=0, top_thresh=0):
	pool = Pool(cpu_count())
	m = Manager()
	queue = m.Queue()  # create queue to save all the results
	tasks = []

	# LEAD

	X = X.reshape(X.shape[0], X.shape[2], X.shape[1])
	avs = [[[] for _ in range(X.shape[1])] for _ in range(X.shape[0])]
	for lead in range(X.shape[0]):
		# take lead, transpose in order to compare every target

		for feature in range(len(X[lead])):
			x = X[lead][feature]
			tasks.append([x, queue, lead, feature])

	pool.map(execute_average, tasks)  # create the results
	for _ in tqdm.tqdm(pool.imap_unordered(execute_average, tasks), total=len(tasks)):
		pass
	while not queue.empty():
		av, lead, feature = queue.get()
		avs[lead][feature] = av
	avs = np.array(avs)
	pool.close()
	pool.join()
	return avs


def execute_average(args):
	x, queue, lead, feature = args[0], args[1], args[2], args[3]
	av = x.mean()
	queue.put((av, lead, feature))


def execute_slope(args):
	f, x, queue, lead, trial, feature = args[0], args[1], args[2], args[3], args[4], args[5]
	m, b = np.polyfit(x, f, 1)
	queue.put((m, lead, trial, feature))


##############################################
## RUN
##############################################

#filenames = ['binned_all_features_002_m.pkl','binned_all_features_002_p.pkl']
#filenames = ['bin_perc_filtered.pkl', 'bin_mem_filtered.pkl']
patient_name = 'raw_FAC007'
segment_patient_data = True
bin_size = 880
with_overlap = False
overlap_step_size = 220
extract_frequency_data = True
frequency_band_mem = 'theta'
frequency_band_perc = 'alpha'
normilise_signal = True
decision_name = 'perc'
pickle_file = './preprocessed/pickle/features_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(decision_name, patient_name, segment_patient_data, bin_size, with_overlap, overlap_step_size, extract_frequency_data, frequency_band_mem, frequency_band_perc, normilise_signal)

#filenames = ['features_mem_raw_FAC002_True_880_False_220_True_theta_alpha.pkl', 'features_perc_raw_FAC002_True_880_False_220_True_theta_alpha.pkl']
#for filename in filenames:

with open(pickle_file, 'rb') as f:
	eeg_m =	pickle.load(f)
feature_names = eeg_m[1]
eeg_m = eeg_m[0]
patient_data = load_raw(patient_name)
if decision_name == 'perc':
	target = patient_data['simVecP'][0:eeg_m.shape[1]]
else:
	target = patient_data['simVecM'][0:eeg_m.shape[1]]

del patient_data

eeg_m, target = filter_features(eeg_m,target, True)

###########
## GET SLOPES
q = multithread_slope_extraction(eeg_m)

if not os.path.exists('./preprocessed/eeg_split'):
	os.makedirs('./preprocessed/eeg_split')
filename = './preprocessed/eeg_split/slopes_features_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(decision_name, patient_name, segment_patient_data, bin_size, with_overlap, overlap_step_size, extract_frequency_data, frequency_band_mem, frequency_band_perc, normilise_signal)
with open(filename, 'wb') as f:
	pickle.dump(q, f)

s = deepcopy(q)
d = deepcopy(q)
del q

###########
## GET AVERAGE SLOPES
# a = multithread_average_slope(s)
# if not os.path.exists('./preprocessed/eeg_split'):
# 	os.makedirs('./preprocessed/eeg_split')
# with open('./preprocessed/eeg_split/average_slopes_' + filename +'.pkl', 'wb') as f:
# 	pickle.dump(a, f)

###########
## GET SLOPE CORRELATION TO TARGET
important = get_importances(d,target)
filename = './preprocessed/eeg_split/importances_features_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(decision_name, patient_name, segment_patient_data, bin_size, with_overlap, overlap_step_size, extract_frequency_data, frequency_band_mem, frequency_band_perc, normilise_signal)
with open(filename, 'wb') as f:
	pickle.dump(important, f)

keepers_all = []
for lead in range(len(important)):
	keepers_feature = []
	for feature in range(len(important[lead])):
		if (important[lead][feature] < 0.42) or  (important[lead][feature] > 0.58):
			keepers_feature.append(feature)
	keepers_all.append(keepers_feature)
with open(pickle_file, 'wb') as f:
	pickle.dump(keepers_all, f)
