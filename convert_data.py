import scipy.io as sio
import pickle

#############################################
## Preperation for this script:
## 1. Clone github reporsitory into folder of your choice
## 2. Download data ( https://unishare.nl/index.php/s/Hk6kYh88aIeVd0Z )
## 3. Extract data into parent folder of where you cloned the gihub repo
## 4. Run this file, it will save new pickles of our data


mat_contents = sio.loadmat('../preprocessed/datathetaOscTLbyTimeV_FAC001.mat')

X_memory = mat_contents['dataMatM']
y_memory = mat_contents['simVecM']

X_perc = mat_contents['dataMatP']
y_perc = mat_contents['simVecP']

with open('../preprocessed/X_memory.pkl', 'wb') as fp:
    pickle.dump(X_memory, fp)
with open('../preprocessed/y_memory.pkl', 'wb') as fp:
    pickle.dump(y_memory, fp)
with open('../preprocessed/X_perc.pkl', 'wb') as fp:
    pickle.dump(X_perc, fp)
with open('../preprocessed/y_perc.pkl', 'wb') as fp:
    pickle.dump(y_perc, fp)
