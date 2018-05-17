from __future__ import division
import scipy.io as sio
import pickle
import numpy as np
import random
import math
from scipy.fftpack import fft, irfft, rfft
from scipy.optimize import curve_fit
import cmath
import matplotlib.pyplot as plt

length = 134
skip = [10,60,61,62,63,64,77,78,79]
eeg_mem = []

print("Creating pickles...")
j = 0
for i in range(length):
	if i <= 78:
		if i+1 == skip[j]:
			j += 1
			continue
	mat_contents = sio.loadmat('FAC002 raw\memory\EEG_DATA_FAC002_MEM_chan_{}.mat'.format(str(i+1)))
	eeg = mat_contents['eeg']
	with open ('FAC002 raw\pickles\eeg_mem_fac2_{}.pkl'.format(str(i+1)), 'wb') as fp:
		pickle.dump(eeg, fp)

# load pickle files
j = 0
for i in range (length):
	if i <= 78:
		if i+1 == skip[j]:
			j += 1
			continue
	with open ('FAC002 raw\pickles\eeg_mem_fac2_{}.pkl'.format(i+1), 'rb') as fp:
		eeg_mem.append(pickle.load(fp))

lengths = []

for i in range(len(eeg_mem[0])):
	plt.plot(eeg_mem[0][i])
plt.show()

boundaries = [[24, 26], [26, 28], [28, 30], [30, 31]]

# Plot leads
def print_leads(num_of_leads, boundaries=None):
	if num_of_leads > len(eeg):
		return False
	for i in range(num_of_leads):
		if boundaries is not None:
			if eeg[i][0] > boundaries[0] and eeg[i][0] < boundaries[1]:
				plt.plot(eeg[i])
		else:
			plt.plot(eeg[i])
	plt.show()
