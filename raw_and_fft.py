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


length = 134 # number of eeg files
skip_mem = [10,60,61,62,63,64,77,78,79] # non existing files
eeg_mem = [] # eeg memory data


ans = input("Have you already created the pickles? (y / n)")
if ans == 'n':
	j = 0
	for i in range(length):
		if i <= skip_mem[-1] - 1:
			if i+1 == skip_mem[j]:
				j += 1
				continue
		mat_contents = sio.loadmat('FAC002 raw\memory\EEG_DATA_FAC002_MEM_chan_{}.mat'.format(str(i+1)))
		eeg = mat_contents['eeg']
		with open ('FAC002 raw\pickles\eeg_mem_fac2_{}.pkl'.format(str(i+1)), 'wb') as fp:
			pickle.dump(eeg, fp)

ans = input("Wanna load pickles? (y / n)")
if ans == 'y':
	j = 0
	for i in range (length):
		if i <= skip_mem[-1] - 1:
			if i+1 == skip_mem[j]:
				j += 1
				continue
		with open ('FAC002 raw\pickles\eeg_mem_fac2_{}.pkl'.format(i+1), 'rb') as fp:
			eeg_mem.append(pickle.load(fp))

eeg_mem = patient_data['eeg_m']
for i in range(len(eeg_mem[0])):
	plt.title("Lead {}".format(1))
	plt.plot(eeg_mem[0][i])
plt.show()


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


def dft_4 (signal, fs=20000, t=1):
	N = int(fs*t)
	y = signal
	yf = fft(y)
	xf = np.linspace(0.0, 0.5*fs, N//2)
	plt.grid()
	plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
	return np.fft.ifft(yf)
	# plt.show()


def cut_eeg(signal, from_point, to_point, lead=0):
	return signal[lead][from_point:to_point]


def mean(lead):
	s = 0
	for i in range(len(lead)):
		s += lead[i]
	return s/len(lead)


def differencing(x):
	out = []
	for i in range(len(x) - 1):
		out.append(x[i+1] - x[i])
	return np.array(out)

lead = 0

# print_certain(differencing(differencing((eeg_mem[0][0]))))

dft_4(differencing(eeg_mem[0][1]), 4400, 1)
dft_4(differencing(eeg_mem[0][2]), 4400, 1)
dft_4(differencing(eeg_mem[1][1]), 4400, 1)
dft_4(differencing(eeg_mem[1][2]), 4400, 1)


lead_name = 2
from_s = 0
to_s = 12000


def show_regions():
	bound = []
	prev = 0
	th = 0.5
	for i in range(len(eeg)):
		diff = abs(mean(eeg[i]) - prev)
		prev = mean(eeg[i])
		if diff > th:
			bound.append(i)
	bound.append(len(eeg))
	print(bound)
	print(len(bound))
	for i in range(len(bound)-1):
		for j in range(bound[i], bound[i+1]):
			dft_3(differencing(cut_eeg(eeg, from_s, to_s, j)))
		# plt.show()


"""
# plt.plot(log_diff(cut_eeg(eeg, from_s, to_s, 1)))
# plt.plot(differencing(cut_eeg(eeg, from_s, to_s)))
# plt.show()

# F = [[0 for x in range(20)] for y in range(leads)]
# for i in range(leads):
# 	F[i] = dft_3(log_diff(cut_eeg(eeg, from_s, to_s, i)))
"""

"""
t = 1
fs = 20000
x = np.linspace(0.0, t, fs*t, endpoint=False)
noise = 5
A1, A2 = 10, 8
f1, f2 = 80, 50
s = A1 * np.sin(2.0 * np.pi * f1 * x) + A2 * np.sin(2.0 * np.pi * f2 * x)
s += [np.random.uniform(-noise, noise) for _ in range(len(x))]
plt.plot(s)
plt.show()
plt.plot(dft_4(s, fs, t))
plt.plot(s)
plt.plot(dft_4(s-[np.random.uniform(-noise, noise) for _ in range(len(x))]) + 10)
"""

"""
signal = eeg[0]
plt.plot(differencing(signal))
th = 0.02
plt.plot(dft_4(differencing(signal), 2000, 6) + th)
"""

plt.show()
