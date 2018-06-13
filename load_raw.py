from __future__ import division
import scipy.io as sio
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, lfilter, filtfilt
import scipy

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
			patient_data['simVecM'] = np.array(patient_data['simVecM'])
			patient_data['simVecP'] = np.array(patient_data['simVecP'])
			with open(preprocessed_location + 'pickle/eegMem_{}.pkl'.format(patient), 'wb') as f:
				pickle.dump(patient_data['eeg_m'], f)
			with open(preprocessed_location + 'pickle/eegPerc_{}.pkl'.format(patient), 'wb') as f:
				pickle.dump(patient_data['eeg_p'], f)
			with open(preprocessed_location + 'pickle/simMem_{}.pkl'.format(patient), 'wb') as f:
				pickle.dump(patient_data['simVecM'], f)
			with open(preprocessed_location + 'pickle/simPerc_{}.pkl'.format(patient), 'wb') as f:
				pickle.dump(patient_data['simVecP'], f)

	patient_data = {}

	patient = patient_name
	print('loading patient {}'.format(patient))

	with open(preprocessed_location + 'pickle/eegMem_{}.pkl'.format(patient), 'rb') as f:
		patient_data['eeg_m'] = pickle.load(f)
	with open(preprocessed_location + 'pickle/eegPerc_{}.pkl'.format(patient), 'rb') as f:
		patient_data['eeg_p'] = pickle.load(f)
	with open(preprocessed_location + 'pickle/simMem_{}.pkl'.format(patient), 'rb') as f:
		patient_data['simVecM'] = pickle.load(f)
	with open(preprocessed_location + 'pickle/simPerc_{}.pkl'.format(patient), 'rb') as f:
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
## Filter signal
############################################################################ '''

def lowpass(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def highpass(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def filter_signal(data, LOW, HIGH, fs, order=5):
    y = lowpass(data, HIGH, fs, order=order)
    y = highpass(y, LOW, fs, order=order)
    return y

# ''' ############################################################################
# ## Test filtering on sine wave
# ############################################################################ '''
#
# n_samples = 4400
# 
#
# fs = 2000.0
# T = n_samples / fs
# t = np.linspace(0, T, n_samples, endpoint=False)
# LOW = 4
# HIGH = 9
#
# x = np.sin(2 * np.pi * 4 * t)
# x += np.sin(2 * np.pi * 9 * t)
# x += np.sin(2 * np.pi * 90 * t)
# x += np.sin(2 * np.pi * 3 * t)
# n=[np.random.randint(100)/10 for i in range(len(t))]
# x+=n
#
# xx =  np.sin(2 * np.pi * 4 * t) + np.sin(2 * np.pi * 9 * t)
#
# y = filter_signal(x, LOW, HIGH, fs)
#
# plt.plot(t, xx, 'b')
# plt.plot(t, y, 'r')
# plt.show()


# eeg_mem = patient_data['eeg_m']
#
# lengths = []
#
# for i in range(len(eeg_mem[0])):
# 	plt.plot(eeg_mem[0][i])
# plt.show()
#
# boundaries = [[24, 26], [26, 28], [28, 30], [30, 31]]
#
# # Plot leads
# def print_leads(num_of_leads, boundaries=None):
# 	if num_of_leads > len(eeg):
# 		return False
# 	for i in range(num_of_leads):
# 		if boundaries is not None:
# 			if eeg[i][0] > boundaries[0] and eeg[i][0] < boundaries[1]:
# 				plt.plot(eeg[i])
# 		else:
# 			plt.plot(eeg[i])
# 	plt.show()
