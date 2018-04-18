import scipy.io as sio
import pickle
import numpy as np

########n#####################################
## Preperation for this script:
## 1. Clone github reporsitory into folder of your choice
## 2. Download data ( https://unishare.nl/index.php/s/Hk6kYh88aIeVd0Z )
## 3. Extract data into parent folder of where you cloned the gihub repo
## 4. Run this file, it will save new pickles of our data

patients = ['FAC001', 'FAC002', 'FAC004', 'FAC005', 'FAC006', 'FAC007', 'FAC008', 'FAC009', 'FAC010', 'FAC011',
			'FAC012', 'FAC013', 'FAC014', 'FAC015', 'FAC016']
for patient in patients:
	mat_contents = sio.loadmat('preprocessed/datathetaOscTLbyTimeV_{}.mat'.format(patient))

	dt_p = mat_contents['NtimePointsP'][0][0]
	dt_m = mat_contents['NtimePointsM'][0][0]

	X_memory = mat_contents['dataMatM']
	y_memory = mat_contents['simVecM']

	X_perc = mat_contents['dataMatP']
	y_perc = mat_contents['simVecP']

	all = []
	trial = []
	# roll over every lead, create new matrix

	for i in range(0, X_memory.shape[0]):
		trial = []
		for n in range(0, len(X_memory[i]) - dt_m + 1, dt_m):
			single = X_memory[i][n:n + dt_m]
			trial.append(single)
		all.append(trial)
	X_memory = np.asarray(all)
	all = []
	trial = []
	# roll over every lead, create new matrix

	for i in range(0, X_perc.shape[0]):
		trial = []
		for n in range(0, len(X_perc[i]) - dt_p + 1, dt_p):
			single = X_perc[i][n:n + dt_p]
			trial.append(single)
		all.append(trial)
	X_perc = np.asarray(all)

	trials = X_memory.shape[0]
	leads = X_memory.shape[1]

	total_mem = [[] for _ in range(leads)]
	for lead in range(0, leads):
		for trial in range(0, trials):
			entry = X_memory[trial][lead]
			total_mem[lead].append(entry)

	trials = X_perc.shape[0]
	leads = X_perc.shape[1]

	total_perc = [[] for _ in range(leads)]
	for lead in range(0, leads):
		for trial in range(0, trials):
			entry = X_perc[trial][lead]
			total_perc[lead].append(entry)
	X_memory, X_perc = np.asarray(total_mem), np.asarray(total_perc)

	with open('preprocessed/{}.pkl'.format(patient), 'wb') as fp:
		pickle.dump([dt_p, dt_m, X_memory, y_memory, X_perc, y_perc], fp)

# To read any patient: [dt_p, dt_m, X_memory, y_memory, X_perc, y_perc] = pickle.load(open('preprocessed/FAC001.pkl', 'rb'))
