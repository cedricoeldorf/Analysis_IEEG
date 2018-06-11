import scipy.io as sio
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

''' ############################################################################
## Load raw
########################################################################### '''

def load_raw(patient_name):
	preprocessed_location = 'preprocessed/'
	# find all the patients that have to be read in...
	preprocessed_dir = next(os.walk(preprocessed_location))[1]
	patients = []
	for dir in preprocessed_dir:
		if 'raw_' in dir:
			patients.append(dir)
	print('{} patient(s) detected'.format(len(patients)))

	if len(patients) != 0 and not os.path.exists(preprocessed_location + 'pickle/eegMem_{}.pkl'.format(patient_name)):
		print('creating pickle for all patients...')
		try:
			os.mkdir(preprocessed_location + 'pickle/')
		except:
			pass
		# create the data for these patients...
		# patient_data = []
		for i, patient in enumerate(patients): # loop over patients
			print(patient)
			patient_dir = preprocessed_location + patient
			patient_data = {}
			mem_dir= patient_dir + '/memory/'
			perc_dir = patient_dir + '/perception/'
			list_o_leads = np.array(os.listdir(mem_dir)) # loop over leads
			lead_name = 'eeg_m' # key for memory eegs
			y_name = 'simVecM' # key for memory y values
			for l in list_o_leads:
				if '_chan_' in l: # read in lead
					mat_contents = sio.loadmat(mem_dir + l)
					eeg = mat_contents['eeg']
					if lead_name not in patient_data:
						patient_data[lead_name] = []
					patient_data[lead_name].append(eeg)
				elif '_simVec' in l: # read in the y values
					mat_contents = sio.loadmat(mem_dir + l)
					y_vec = mat_contents[y_name]
					patient_data[y_name] = y_vec

			list_o_leads = np.array(os.listdir(perc_dir))  # loop over leads
			lead_name = 'eeg_p'  # key for perception eegs
			y_name = 'simVecP'  # key for perception y values
			for l in list_o_leads:
				if '_chan_' in l:  # read in lead
					mat_contents = sio.loadmat(perc_dir + l)
					eeg = mat_contents['eeg']
					# print(patient, l, eeg.shape)
					if lead_name not in patient_data:
						patient_data[lead_name] = []
					patient_data[lead_name].append(eeg)
				elif '_simVec' in l:  # read in the y values
					mat_contents = sio.loadmat(perc_dir + l)
					y_vec = mat_contents[y_name]
					patient_data[y_name] = y_vec
		# Now patient data contains all the information we need
			patient_data['eeg_m'] = np.array(patient_data['eeg_m'])
			patient_data['eeg_p'] = np.array(patient_data['eeg_p'])
			if 'simVecM' in patient_data:
				patient_data['simVecM'] = np.array(patient_data['simVecM'])
				patient_data['simVecP'] = np.array(patient_data['simVecP'])
			with open(preprocessed_location + 'pickle/eegMem_{}.pkl'.format(patient), 'wb') as f:
				pickle.dump(patient_data['eeg_m'], f)
			with open(preprocessed_location + 'pickle/eegPerc_{}.pkl'.format(patient), 'wb') as f:
				pickle.dump(patient_data['eeg_p'], f)
			if 'simVecM' in patient_data:
				with open(preprocessed_location + 'pickle/simMem_{}.pkl'.format(patient), 'wb') as f:
					pickle.dump(patient_data['simVecM'], f)
				with open(preprocessed_location + 'pickle/simPerc_{}.pkl'.format(patient), 'wb') as f:
					pickle.dump(patient_data['simVecP'], f)

	patient_data = {}

	patient = patient_name
	print('loading patient {}'.format(patient))

	with open(preprocessed_location + 'pickle/eegMem_raw_{}.pkl'.format(patient), 'rb') as f:
		patient_data['eeg_m'] = pickle.load(f)
	with open(preprocessed_location + 'pickle/eegPerc_raw_{}.pkl'.format(patient), 'rb') as f:
		patient_data['eeg_p'] = pickle.load(f)
	with open(preprocessed_location + 'pickle/simMem_raw_{}.pkl'.format(patient), 'rb') as f:
		patient_data['simVecM'] = pickle.load(f)
	with open(preprocessed_location + 'pickle/simPerc_raw_{}.pkl'.format(patient), 'rb') as f:
		patient_data['simVecP'] = pickle.load(f)

	'''
		patient_data = contains all the data for patient X

		patient_data['eeg_m'] = all the memory eeg leads
		patient_data['eeg_p'] = all the perc eeg leads

		patient_data['simVecM'] # all the memory y values
		patient_data['simVecP'] # all the perception y values
	'''
	return patient_data

''' ############################################################################
## Load pickle
########################################################################### '''

def load_pickle(patient_name):

	patient_data = {}
	preprocessed_location = 'preprocessed/'
	patient = patient_name
	print('loading patient {}'.format(patient))

	with open(preprocessed_location + 'pickle/eegMem_raw_{}.pkl'.format(patient), 'rb') as f:
		patient_data['eeg_m'] = pickle.load(f)
	with open(preprocessed_location + 'pickle/eegPerc_raw_{}.pkl'.format(patient), 'rb') as f:
		patient_data['eeg_p'] = pickle.load(f)
	with open(preprocessed_location + 'pickle/simMem_raw_{}.pkl'.format(patient), 'rb') as f:
		patient_data['simVecM'] = pickle.load(f)
	with open(preprocessed_location + 'pickle/simPerc_raw_{}.pkl'.format(patient), 'rb') as f:
		patient_data['simVecP'] = pickle.load(f)

	return patient_data

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

''' ############################################################################
## Segment trials
########################################################################### '''

def segment_lead_matrix(lead_matrix, bin_size):

	n_trials = lead_matrix.shape[0]
	trial_length = lead_matrix.shape[1]
	if trial_length % bin_size != 0:
		print('Wrong bin size, try:')
		print(find_bin_size(trial_length))
		return
	step_size = int(trial_length / bin_size)
	lead_list = []
	for trial in range(n_trials):
		bin = []
		for i in range(bin_size):
			segment = lead_matrix[trial][i:i+step_size]
			bin.append(segment)
		lead_list.append(bin)
	return lead_list

''' ############################################################################
## EX
########################################################################### '''

patient_data = load_pickle('FAC002')
eeg_m = patient_data['eeg_m']
eeg_p = patient_data['eeg_p']
y_m = patient_data['simVecM']
y_p = patient_data['simVecP']

separation_mem = separate_leads(eeg_m)
separation_perc = separate_leads(eeg_p)
mem_leads = []
perc_leads = []
n_leads = len(separation_mem)
bin_size = 40
for i in range(n_leads):
	mem_leads.append(segment_lead_matrix(separation_mem[i], bin_size))
	perc_leads.append(segment_lead_matrix(separation_perc[i], bin_size))
