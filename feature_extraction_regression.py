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


def get_importances(slopes, target):
	#############################
	## Get coherence with target
	#############################
	slopes = slopes.reshape(slopes.shape[0], slopes.shape[2], slopes.shape[1])

	for lead in range(slopes.shape[0]):
		for feature in range(slopes.shape[1]):
			for trial in range(slopes.shape[2]):
				if slopes[lead][feature][trial] < 0:
					slopes[lead][feature][trial] = -1
				else:
					slopes[lead][feature][trial] = 1

	importance = []
	# iterate over leads
	for z in range(slopes.shape[0]):
		acc = []
		for x in range(slopes.shape[1]):
			acc.append(accuracy_score(target, slopes[z][x]))

		importance.append(acc)

	return importance


'''######################################
## attempt to multithread get_slopes function '''


def multithread_slope_extraction(X):
	pool = Pool(cpu_count())

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

with open('./preprocessed/eeg_split/bin_mem.pkl', 'rb') as f:
	eeg_m = pickle.load(f)
feature_names = eeg_m[1]
eeg_m = eeg_m[0]
patient_data = load_raw('raw_FAC002')
target = patient_data['simVecM'][0:eeg_m.shape[1]]
del patient_data

###########
## GET SLOPES
q = multithread_slope_extraction(eeg_m)

if not os.path.exists('./preprocessed/eeg_split'):
	os.makedirs('./preprocessed/eeg_split')
with open('./preprocessed/eeg_split/slopes_eeg_m.pkl', 'wb') as f:
	pickle.dump(q, f)

s = deepcopy(q)
d = deepcopy(q)
del q

###########
## GET AVERAGE SLOPES
a = multithread_average_slope(s)
if not os.path.exists('./preprocessed/eeg_split'):
	os.makedirs('./preprocessed/eeg_split')
with open('./preprocessed/eeg_split/average_slopes_eeg_m.pkl', 'wb') as f:
	pickle.dump(a, f)

###########
## GET SLOPE CORRELATION TO TARGET
important = get_importances(d, target)
with open('./preprocessed/eeg_split/important_slopes_eeg_m.pkl', 'wb') as f:
	pickle.dump(important, f)
