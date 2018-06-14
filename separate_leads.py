import numpy as np
from itertools import islice
from multiprocessing import Pool, Manager, cpu_count
import os, psutil
from extract_statistics import *
import pickle
import os

''' ############################################################################
## Seprate leads
########################################################################### '''


def separate_leads(data):
	n_leads = data.shape[0]
	lead_list = []
	for i in range(n_leads):
		lead_list.append(data[i])
	return lead_list


''' ############################################################################
## Find bin size
########################################################################### '''


def find_bin_size(length):
	list = []
	for i in range(1, length):
		if length % i == 0:
			list.append(i)
	return list


def find_best_bin_size(size, bin):
	for i in range(bin, 1, -1):
		if size % i == 0:
			return i
	return None


''' ############################################################################
## Find overlap size
########################################################################### '''


def find_overlap_size(length, window):
	list = []
	for i in range(1, length):
		if i == window:
			break
		if length % (window - i) == 0:
			list.append(i)
	return list


''' ############################################################################
## Segment trials
########################################################################### '''


def window(signal, window_size=4, step_size=2):
	sh = (signal.size - window_size + 1, window_size)
	st = signal.strides * 2
	view = np.lib.stride_tricks.as_strided(signal, strides=st, shape=sh)[0::step_size]
	return view


def execute(args):
	p = psutil.Process(os.getpid())
	p.nice(10)  # set

	signal, queue, bin_size, overlap, overlap_step, lead, trial = args[0], args[1], args[2], args[3], args[4], args[5], args[6]
	if overlap:
		splitted = window(signal, bin_size, overlap_step)
	else:
		splitted = np.array(np.split(signal, int(len(signal) / bin_size)))
	queue.put((lead, trial, bin_size, overlap, overlap_step, splitted))


def segment_multithreaded_eeg(eeg, bin_size, overlap, overlap_step, n_jobs=-1):
	n_leads = eeg.shape[0]
	n_trials = eeg.shape[1]
	tasks = []
	if n_jobs == -1:
		pool = Pool(cpu_count())
	else:
		pool = Pool(n_jobs)
	manager = Manager()
	queue = manager.Queue()

	for trial in range(n_trials):
		for lead in range(n_leads):
			signal = eeg[lead][trial]
			tasks.append([signal, queue, bin_size, overlap, overlap_step, lead, trial])
	pool.map(execute, tasks)  # create the results
	result_eeg = [[[] for _ in range(n_trials)] for _ in range(n_leads)]
	del tasks
	while not queue.empty():
		(lead, trial, bin_size, overlap, overlap_step, splitted) = queue.get()
		result_eeg[lead][trial] = splitted
	result_eeg = np.array(result_eeg)
	pool.close()
	pool.join()
	# print(result_eeg.shape)
	return result_eeg


## For single threaded use...
def segment_eeg(eeg, bin_size, overlap, overlap_step, separate_bands=False):
	n_leads = eeg.shape[0]
	n_trials = eeg.shape[1]
	if separate_bands:
		# result_eeg = [[[[] for _ in range(len(eeg_bands))] for _ in range(n_trials)] for _ in range(n_leads)]
		print('not implemented')
	else:
		result_eeg = [[[] for _ in range(n_trials)] for _ in range(n_leads)]
	if overlap:
		if separate_bands:
			for trial in range(n_trials):
				for lead in range(n_leads):
					print('not implemented')
		else:
			for trial in range(n_trials):
				for lead in range(n_leads):
					signal = eeg[lead][trial]
					splitted = np.array(list(zip(*[islice(signal, i * overlap_step, None) for i in range(bin_size)])))
					result_eeg[lead][trial] = splitted
	else:
		if separate_bands:
			for trial in range(n_trials):
				for lead in range(n_leads):
					print('not implemented')
		else:
			for trial in range(n_trials):
				for lead in range(n_leads):
					signal = eeg[lead][trial]
					# if len(signal) % bin_size != 0:
					# 	raise Exception('Wrong bin size {} for lead'.format(bin_size), lead, 'and trial', trial)
					splitted = np.split(signal, int(len(signal) / bin_size))
					result_eeg[lead][trial] = splitted
	result_eeg = np.array(result_eeg)
	return result_eeg


def segments_patient(patient_data, bin_size=200, overlap=False, overlap_step=50, multithreaded=False, n_jobs=-1):
	"""
	bin_size : the size of the bins for windowing, if it doensn't divide nicely it doesn't crash but just have approx equal sized bins
	overlap : whether the bins should overlap
	overlap_step : how big a step for the overlap
	multithreaded : whether to use multi threading
	"""

	if multithreaded:
		patient_data['eeg_m'] = segment_multithreaded_eeg(patient_data['eeg_m'], bin_size, overlap, overlap_step, n_jobs)
		patient_data['eeg_p'] = segment_multithreaded_eeg(patient_data['eeg_p'], bin_size, overlap, overlap_step, n_jobs)
	else:
		patient_data['eeg_m'] = segment_eeg(patient_data['eeg_m'], bin_size, overlap, overlap_step)
		patient_data['eeg_p'] = segment_eeg(patient_data['eeg_p'], bin_size, overlap, overlap_step)
	return patient_data


def segment_lead_matrix(lead_matrix, bin_size, overlap=False, overlap_step=10, include_last_window=True):
	n_trials = lead_matrix.shape[0]
	trial_length = lead_matrix.shape[1]

	if overlap:
		step_size = int(trial_length / bin_size)
		lead_list = []
		for trial in range(n_trials):
			bin = []
			i = 0
			for j in range(0, trial_length):
				if i + bin_size >= trial_length:
					if include_last_window:
						bin.append(lead_matrix[trial][i:])
					break
				segment = lead_matrix[trial][i:i + bin_size]
				bin.append(segment)
				i = i + (bin_size - overlap_step)
			lead_list.append(bin)
	else:
		if trial_length % bin_size != 0 or bin_size > trial_length:
			print('Wrong bin size, try:')
			print(find_bin_size(trial_length))
			return
		step_size = int(trial_length / bin_size)
		step_size = bin_size
		lead_list = []
		for trial in range(n_trials):
			bin = []
			i = 0
			for j in range(trial_length / bin_size):
				segment = lead_matrix[trial][i:i + step_size]
				bin.append(segment)
				i += step_size
			lead_list.append(bin)

	return lead_list


''' ############################################################################
## EX
########################################################################### '''

from load_raw import load_raw

input("Hit enter to create binned sample data")
patient_data = load_raw('raw_FAC002')

patient_data = segments_patient(patient_data, 200, overlap=False)

eeg_m = patient_data['eeg_m']
eeg_p = patient_data['eeg_p']
y_m = patient_data['simVecM']
y_p = patient_data['simVecP']

eeg_m = eeg_m.reshape(eeg_m.shape[1], eeg_m.shape[0], 22, 200)
eeg_m = eeg_m[0:10]
eeg_m = eeg_m.reshape(eeg_m.shape[1], eeg_m.shape[0], 22, 200)
eeg_m = eeg_m[0:10]
binned_m = extract_multithreaded_basic(eeg_m)

if not os.path.exists('./preprocessed/eeg_split'):
	os.makedirs('./preprocessed/eeg_split')
with open('./preprocessed/eeg_split/bin_mem.pkl', 'wb') as f:
	pickle.dump(binned_m, f)

#
# separation_mem = separate_leads(eeg_m)
# separation_perc = separate_leads(eeg_p)
# mem_leads = []
# perc_leads = []
# n_leads_m = len(separation_mem)
# n_leads_p = len(separation_perc)
# bin_size = 800
# overlap_size = 15
# for i in range(n_leads_m):
# 	mem_leads.append(segment_lead_matrix(separation_mem[i], bin_size, overlap=True))
# for i in range(n_leads_p):
# 	perc_leads.append(segment_lead_matrix(separation_perc[i], bin_size))
