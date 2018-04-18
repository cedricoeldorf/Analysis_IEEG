import scipy.io as sio
import pickle
import numpy as np

########n#####################################
## Preperation for this script:
## 1. Clone github reporsitory into folder of your choice
## 2. Download data ( https://unishare.nl/index.php/s/Hk6kYh88aIeVd0Z )
## 3. Extract data into parent folder of where you cloned the gihub repo
## 4. Run this file, it will save new pickles of our data

patients = ['FAC001', 'FAC002', 'FAC004', 'FAC005', 'FAC006', 'FAC007', 'FAC008', 'FAC009', 'FAC010', 'FAC011', 'FAC012', 'FAC013', 'FAC014', 'FAC015', 'FAC016']
for patient in patients:
	mat_contents = sio.loadmat('preprocessed/datathetaOscTLbyTimeV_{}.mat'.format(patient))

	dt_p = mat_contents['NtimePointsP'][0][0]
	dt_m = mat_contents['NtimePointsM'][0][0]

	X_memory = mat_contents['dataMatM']
	y_memory = mat_contents['simVecM']

	X_perc = mat_contents['dataMatP']
	y_perc = mat_contents['simVecP']

	with open('preprocessed/{}.pkl'.format(patient), 'wb') as fp:
		pickle.dump([dt_p, dt_m, X_memory, y_memory, X_perc, y_perc], fp)


# To read any patient: [dt_p, dt_m, X_memory, y_memory, X_perc, y_perc] = pickle.load(open('preprocessed/FAC001.pkl', 'rb'))
