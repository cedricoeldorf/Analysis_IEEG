import scipy.io as sio
import pickle
import numpy as np
import os
from scipy.signal import butter, lfilter, filtfilt
from multiprocessing import Pool, Manager, cpu_count
import tqdm, psutil


theta = [4, 9]
beta = [14, 28]
alpha = [9, 14]
delta = [2, 4]
low_gamma = [28, 48]
high_gamma = [48, 90]


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
		for i, patient in enumerate(patients):  # loop over patients
			print(patient)
			patient_dir = preprocessed_location + patient
			patient_data = {}
			mem_dir = patient_dir + '/memory/'
			perc_dir = patient_dir + '/perception/'
			list_o_leads = np.array(os.listdir(mem_dir))  # loop over leads
			lead_name = 'eeg_m'  # key for memory eegs
			y_name = 'simVecM'  # key for memory y values
			for l in list_o_leads:
				if '_chan_' in l:  # read in lead
					mat_contents = sio.loadmat(mem_dir + l)
					eeg = mat_contents['eeg']
					if lead_name not in patient_data:
						patient_data[lead_name] = []
					patient_data[lead_name].append(eeg)
				elif '_simVec' in l:  # read in the y values
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


def differencing(x):
	return np.diff(x)


def lowpass(data, cutoff, fs, order=5):
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='low', analog=False)
	y = lfilter(b, a, data)
	return y


def highpass(data, cutoff, fs, order=5):
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='high', analog=False)
	y = filtfilt(b, a, data)
	return y


def filter_signal(data, band, fs=2000.0, order=5):
	data = differencing(data)
	y = lowpass(data, band[1], fs, order=order)
	y = highpass(y, band[0], fs, order=order)
	y = np.append(y, [y[-1]])
	return y


def loop(eeg, filter, multithreaded=False, queue=None):
	if multithreaded:
		tasks = []
		if len(eeg.shape) == 4:
			n_leads = eeg.shape[0]
			n_trials = eeg.shape[1]
			n_bins = eeg.shape[2]
			for lead in range(n_leads):
				for trial in range(n_trials):
					for bin in range(n_bins):
						signal = eeg[lead][trial][bin]
						tasks.append([signal, queue, lead, trial, bin, filter])
						# result_features[lead][trial][bin] = filter_signal(signal, filter)
		else:
			n_leads = eeg.shape[0]
			n_trials = eeg.shape[1]
			for lead in range(n_leads):
				for trial in range(n_trials):
						signal = eeg[lead][trial]
						tasks.append([signal, queue, lead, trial, -1, filter])
		return tasks
	else:
		if len(eeg.shape) == 4:
			n_leads = eeg.shape[0]
			n_trials = eeg.shape[1]
			n_bins = eeg.shape[2]
			result_features = [[[[] for _ in range(n_bins)] for _ in range(n_trials)] for _ in range(n_leads)]
			for lead in range(n_leads):
				for trial in range(n_trials):
					for bin in range(n_bins):
						signal = eeg[lead][trial][bin]
						result_features[lead][trial][bin] = filter_signal(signal, filter)
		else:
			n_leads = eeg.shape[0]
			n_trials = eeg.shape[1]
			result_features = [[[] for _ in range(n_trials)] for _ in range(n_leads)]
			for lead in range(n_leads):
				for trial in range(n_trials):
						signal = eeg[lead][trial]
						result_features[lead][trial] = filter_signal(signal, filter)
		return np.array(result_features)


def execute(args):
	try:
		p = psutil.Process(os.getpid())
		p.nice(10)  # set
	except:
		pass
	signal, queue, lead, trial, bin, filter = args[0], args[1], args[2], args[3], args[4], args[5]
	signal = filter_signal(signal, filter)
	queue.put((lead, trial, bin, signal))


def extract_frequency(patient_data, freq_band_m='theta', freq_band_p='alpha', multithreaded=True):
	if multithreaded:
		pool = Pool(cpu_count())
		m = Manager()
		queue = m.Queue()
		tasks = loop(patient_data['eeg_m'], theta, multithreaded=multithreaded, queue=queue)
		for _ in tqdm.tqdm(pool.imap_unordered(execute, tasks), total=len(tasks)):
			pass
		if len(patient_data['eeg_m'].shape) == 4:
			result_features = [[[[] for _ in range(patient_data['eeg_m'].shape[2])] for _ in range(patient_data['eeg_m'].shape[1])] for _ in range(patient_data['eeg_m'].shape[0])]
			while not queue.empty():
				lead, trial, bin, signal = queue.get()
				result_features[lead][trial][bin] = signal
		else:
			result_features = [[[] for _ in range(patient_data['eeg_m'].shape[1])] for _ in range(patient_data['eeg_m'].shape[0])]
			while not queue.empty():
				lead, trial, bin, signal = queue.get()
				result_features[lead][trial] = signal
		patient_data['eeg_m'] = np.array(result_features)

		print('done with memory signal extraction...')
		del queue
		queue = m.Queue()
		tasks = loop(patient_data['eeg_p'], alpha, multithreaded=multithreaded, queue=queue)
		for _ in tqdm.tqdm(pool.imap_unordered(execute, tasks), total=len(tasks)):
			pass
		if len(patient_data['eeg_p'].shape) == 4:
			result_features = [[[[] for _ in range(patient_data['eeg_p'].shape[2])] for _ in range(patient_data['eeg_p'].shape[1])] for _ in range(patient_data['eeg_p'].shape[0])]
			while not queue.empty():
				lead, trial, bin, signal = queue.get()
				result_features[lead][trial][bin] = signal
		else:
			result_features = [[[] for _ in range(patient_data['eeg_p'].shape[1])] for _ in range(patient_data['eeg_p'].shape[0])]
			while not queue.empty():
				lead, trial, bin, signal = queue.get()
				result_features[lead][trial] = signal
		patient_data['eeg_p'] = np.array(result_features)
		print('done with perception signal extraction...')
		pool.close()
		pool.join()
		del pool, queue, m, tasks
	else:
		if freq_band_m == 'theta':
			patient_data['eeg_m'] = loop(patient_data['eeg_m'], theta)
		elif freq_band_m == 'beta':
			patient_data['eeg_m'] = loop(patient_data['eeg_m'], beta)
		elif freq_band_m == 'alpha':
			patient_data['eeg_m'] = loop(patient_data['eeg_m'], alpha)
		elif freq_band_m == 'delta':
			patient_data['eeg_m'] = loop(patient_data['eeg_m'], delta)
		elif freq_band_m == 'low_gamma':
			patient_data['eeg_m'] = loop(patient_data['eeg_m'], low_gamma)
		elif freq_band_m == 'high_gamma':
			patient_data['eeg_m'] = loop(patient_data['eeg_m'], high_gamma)
		else:
			raise Exception('invalid freq band for mem!')
		print('done with memory signal extraction...')

		if freq_band_p == 'theta':
			patient_data['eeg_p'] = loop(patient_data['eeg_p'], theta)
		elif freq_band_p == 'beta':
			patient_data['eeg_p'] = loop(patient_data['eeg_p'], beta)
		elif freq_band_p == 'alpha':
			patient_data['eeg_p'] = loop(patient_data['eeg_p'], alpha)
		elif freq_band_p == 'delta':
			patient_data['eeg_p'] = loop(patient_data['eeg_p'], delta)
		elif freq_band_p == 'low_gamma':
			patient_data['eeg_p'] = loop(patient_data['eeg_p'], low_gamma)
		elif freq_band_p == 'high_gamma':
			patient_data['eeg_p'] = loop(patient_data['eeg_p'], high_gamma)
		else:
			raise Exception('invalid freq band for perc!')
		print('done with perception signal extraction...')

	return patient_data


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
