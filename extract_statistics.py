import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import variation



####################################
## Extract statistics for every lead and create AV table
####################################

def extract_basic(X):
	print("extracting basics")
	all = []  # will be the whole dataset
	p = 50  # for generalized mean
	# Iterate over every trial
	for i in range(0, X.shape[1]):
		small = []  # this is temporary list to add to new data set after every iteration
		feature_names = []  # for later feature extraction, we create a list of names

		# get every lead for current trial
		for j in range(0, X.shape[0]):
			signal = X[j][i]
			vertices, smoothed_signal = create_vertex2vertex(signal,spacing=10)

			########
			## mean
			small.append(signal.mean())
			feature_names.append("mean_lead_" + str(j + 1))
			########
			## Max
			small.append(signal.max())
			feature_names.append("max_lead_" + str(j + 1))
			########
			## Min
			small.append(signal.min())
			feature_names.append("min_lead_" + str(j + 1))
			########
			## RMS
			small.append(RMS(signal))
			feature_names.append("rms_lead_" + str(j + 1))
			########
			## harmonic
			small.append(harmonic(signal))
			feature_names.append("harmonic_lead_" + str(j + 1))
			########
			## geometric
			small.append(geometric(signal))
			feature_names.append("geometric_lead_" + str(j + 1))
			########
			## generalized
			small.append(generalized_mean(signal, p))
			feature_names.append("generalized_lead_" + str(j + 1))

			########
			## Piecewise Aggregate Approximation
			## (split series into parts and take mean of each0)
			## This makes sense as the neurons should be firing in aggregating f,)
			m1, m2, m3 = PAA(X)
			small.append(m1)
			feature_names.append("PAA1_" + str(j + 1))
			small.append(m2)
			feature_names.append("PAA2_" + str(j + 1))
			small.append(m3)
			feature_names.append("PAA3_" + str(j + 1))

		# all.append(small)
	all = np.asarray(all)
	return all, feature_names


def cuvv_nm_std_cov(signal, vertices): # mean and S.D. coefficient of variation of curvatures at vertices
	dx_dt = np.gradient(vertices)
	dy_dt = np.gradient(signal[vertices])
	d2x_dt2 = np.gradient(dx_dt)
	d2y_dt2 = np.gradient(dy_dt)
	curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
	return np.mean(curvature), np.std(curvature), variation(curvature)


def create_vertex2vertex(data, spacing=10, smooth=True, window=51, polyorder=3):
	"""Finds peaks and valley in `data` which are of `spacing` width.
	:param polyorder: degree of polynomial to fit
	:param window: size of windows for smoothing
	:param smooth: whether to smooth the signal or not
	:param data: values
	:param spacing: minimum spacing to the next peak (should be 1 or more)
	:return: list of indices that are peaks and valleys
	"""
	if smooth:
		data = savgol_filter(data, window, polyorder)
	len = data.size
	x = np.zeros(len + 2 * spacing)
	x[:spacing] = data[0] - 1.e-6
	x[-spacing:] = data[-1] - 1.e-6
	x[spacing:spacing + len] = data
	peak_candidate = np.zeros(len)
	valley_candidate = np.zeros(len)
	peak_candidate[:] = True
	valley_candidate[:] = True
	for s in range(spacing):
		start = spacing - s - 1
		h_b = x[start: start + len]  # before
		start = spacing
		h_c = x[start: start + len]  # central
		start = spacing + s + 1
		h_a = x[start: start + len]  # after
		peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))
		valley_candidate = np.logical_and(valley_candidate, np.logical_and(h_c < h_b, h_c < h_a))

	ind_peak = np.argwhere(peak_candidate)
	ind_valley = np.argwhere(valley_candidate)
	ind_peak = ind_peak.reshape(ind_peak.size).tolist()
	ind_valley = ind_valley.reshape(ind_valley.size).tolist()
	ind = set(ind_peak+ind_valley)
	ind = list(ind)
	ind.sort()
	return ind, data


def RMS(lead):
	sum = 0
	for i in range(len(lead)):
		sum += lead[i] ** 2
	return np.sqrt(sum / len(lead))


def harmonic(lead):
	sum = 0
	for i in range(len(lead)):
		sum += 1 / lead[i]
	return len(lead) / sum


def geometric(lead):
	sum = 1
	for i in range(len(lead)):
		if lead[i] != 0:
			sum *= lead[i]
	return abs(sum) ** (1 / len(lead))


def generalized_mean(lead, p):
	sum = 1
	for i in range(len(lead)):
		sum *= lead[i]
	return abs(sum) ** (1 / p)


## Piecewise Aggregate Approximation
def PAA(X, split=3):
	length = int(len(X) / split)
	m1 = X[:length]
	m2 = X[length:length + length]
	m3 = X[length + length:]

	m1 = m1.mean()
	m2 = m2.mean()
	m3 = m3.mean()
	return m1, m2, m3