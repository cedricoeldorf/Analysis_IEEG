import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import variation
from scipy.ndimage.interpolation import shift


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
			# vertices, smoothed_signal = create_vertex2vertex(signal,spacing=10)
			'''
			print(vertices) the positions in smoothed signal that were selected as vertices
			print(smoothed_signal[vertices]) the corresponding value for these positions
			'''
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


def curvature_period_features(signal, AMSD):  # mean and S.D. coefficient of variation of curvatures at vertices
	"""
			mean of curvatures (d2x/dt2) at vertices
			S.D. of curvatures at vertices
			coefficient of variation of curvatures at vertices
			vertex counts/sec
			S.D. of vertex-to-vertex period
			coefficient of variation of vertex-to-vertex period
			count of mean crossings/sec (hysteresis = 25% of AMSD)
	"""
	vertices, signal_smooth = create_vertex2vertex(signal, spacing=10)

	dx_dt = np.gradient(vertices)
	dy_dt = np.gradient(signal_smooth[vertices])
	d2x_dt2 = np.gradient(dx_dt)
	d2y_dt2 = np.gradient(dy_dt)
	curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
	seconds = len(signal_smooth) / 2000.0  # 2000hz sample rate and signal is 4400 samples
	vertices_per_second = len(signal_smooth[vertices] / seconds)
	selected_period = np.array(vertices[::2])
	vertices_period = np.subtract(selected_period[1::], selected_period[:-1:])
	hysteresis = abs(0.25 * AMSD)
	mean = signal_smooth.mean()
	shifted_signal = signal_smooth - mean
	hysterisized_signal = hyst(shifted_signal, -hysteresis, hysteresis)
	zero_crossing_count = (np.diff(hysterisized_signal) != 0).sum()
	CTMXMN = zero_crossing_count / seconds

	return curvature.mean(), \
		   curvature.std(), \
		   variation(curvature), \
		   vertices_per_second, \
		   vertices_period.std(), \
		   variation(vertices_period), \
		   CTMXMN


def hyst(x, th_lo, th_hi, initial=False):
	"""
	x : Numpy Array
		Series to apply hysteresis to.
	th_lo : float or int
		Below this threshold the value of hyst will be False (0).
	th_hi : float or int
		Above this threshold the value of hyst will be True (1).
	"""

	if th_lo > th_hi:  # If thresholds are reversed, x must be reversed as well
		x = x[::-1]
		th_lo, th_hi = th_hi, th_lo
		rev = True
	else:
		rev = False

	hi = x >= th_hi
	lo_or_hi = (x <= th_lo) | hi

	ind = np.nonzero(lo_or_hi)[0]
	if not ind.size:  # prevent index error if ind is empty
		x_hyst = np.zeros_like(x, dtype=bool) | initial
	else:
		cnt = np.cumsum(lo_or_hi)  # from 0 to len(x)
		x_hyst = np.where(cnt, hi[ind[cnt - 1]], initial)

	if rev:
		x_hyst = x_hyst[::-1]

	return x_hyst


def create_vertex2vertex(data, spacing=10, smooth=True, window=51, polyorder=3):
	## Input: Single signal array
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
	ind = set(ind_peak + ind_valley)
	ind = list(ind)
	ind.sort()
	return ind, data


# Feature #27
def crest(signal):
	""" Feature #27

		Returns: Maximum peak-to-peak amplitude / AMSD
				 (So called crest factor)
	"""
	# Calculate RMS
	rms = RMS(signal)
	# Get max
	max = np.max(signal)
	# Return ratio
	return max / rms


# Feature #28
def RAPN(signal):
	""" Feature #28

		Returns: Mean of positive amplitudes / mean of negative amplitudes
	"""
	# Get max indices
	# max_indices = peakutils.indexes(signal)
	max_indices = peakutils.indexes(signal, thres=0.02 / max(signal), min_dist=0.1)
	# Get min indices
	# min_indices = peakutils.indexes(signal*(-1))
	min_indices = peakutils.indexes(signal * (-1), thres=0.02 / max(signal * (-1)), min_dist=0.1)
	# Mean of positive amplitudes
	mean_of_pos = np.mean([signal[i] for i in max_indices])
	# Mean of negative amplitudes
	mean_of_neg = np.mean([signal[i] for i in min_indices])
	# return ratio
	return mean_of_pos / mean_of_neg


# Feature #29
def RTRF(signal):
	""" Feature #29

		Returns: Mean rise time / mean fall time
	"""
	# Get max indices
	max_indices = peakutils.indexes(signal, thres=0.02 / max(signal), min_dist=0.1)
	# Get min indices
	min_indices = peakutils.indexes(signal * (-1), thres=0.02 / max(signal * (-1)), min_dist=0.1)
	# Extract rise times and fall times
	rise_time = []
	fall_time = []
	if max_indices[0] > min_indices[0]:
		rise_time.append([max_indices[i] - min_indices[i] for i in range(min(len(max_indices), len(min_indices)) - 1)])
		fall_time.append([min_indices[i + 1] - max_indices[i] for i in range(min(len(max_indices), len(min_indices)) - 1)])
	else:
		rise_time.append([max_indices[i + 1] - min_indices[i] for i in range(min(len(max_indices), len(min_indices)) - 1)])
		fall_time.append([min_indices[i] - max_indices[i] for i in range(min(len(max_indices), len(min_indices)) - 1)])
	# Mean of rise and fall time
	rise_mean = np.sum(rise_time) / len(rise_time[0])
	fall_mean = np.sum(fall_time) / len(fall_time[0])
	# Return ratio
	return rise_mean / fall_mean


def RTPN(signal):
	""" Feature # 30

		Returns: Mean period of crossings of the mean positive amplitude /
				 mean period of crossings of the mean negative amplitude
	"""
	# Get max indices
	# max_indices = peakutils.indexes(signal)
	max_indices = peakutils.indexes(signal, thres=0.02 / max(signal), min_dist=0.1)
	# Get min indices
	# min_indices = peakutils.indexes(signal*(-1))
	min_indices = peakutils.indexes(signal * (-1), thres=0.02 / max(signal * (-1)), min_dist=0.1)
	# Mean of positive amplitudes
	mean_of_pos = np.mean([signal[i] for i in max_indices])
	# Mean of negative amplitudes
	mean_of_neg = np.mean([signal[i] for i in min_indices])
	all_mean_pos_crossing = []
	all_mean_neg_crossing = []
	pos_mean_cross_up = False
	pos_mean_cross_down = False
	neg_mean_cross_up = False
	neg_mean_cross_down = False
	for i in range(len(signal) - 1):

		# Cross positive mean up
		if signal[i + 1] > mean_of_pos and signal[i] < mean_of_pos:
			pos_mean_cross_up = i
		# Cross positive mean down
		if signal[i + 1] < mean_of_pos and signal[i] > mean_of_pos:
			pos_mean_cross_down = i
		# Append to positive crossing list
		if not isinstance(pos_mean_cross_up, bool):
			all_mean_pos_crossing.append(pos_mean_cross_down - pos_mean_cross_up)
			pos_mean_cross_up = False
			pos_mean_cross_down = False
		# Cross negative mean down
		if signal[i + 1] < mean_of_neg and signal[i] > mean_of_neg:
			neg_mean_cross_down = i
		# Cross negative mean up
		if signal[i + 1] > mean_of_neg and signal[i] < mean_of_neg:
			neg_mean_cross_up = i
		# Append to negative crossing list
		if not isinstance(neg_mean_cross_down, bool):
			all_mean_neg_crossing.append(neg_mean_cross_up - neg_mean_cross_down)
			neg_mean_cross_up = False
			neg_mean_cross_down = False
	# Get means
	mean_pos_crossing = np.mean(all_mean_pos_crossing)
	mean_neg_crossing = np.mean(all_mean_neg_crossing)
	# Return ratio
	return mean_pos_crossing / mean_neg_crossing


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


#  Piecewise Aggregate Approximation
def PAA(X, split=3):
	length = int(len(X) / split)
	m1 = X[:length]
	m2 = X[length:length + length]
	m3 = X[length + length:]

	m1 = m1.mean()
	m2 = m2.mean()
	m3 = m3.mean()
	return m1, m2, m3


def amplitude_features(signal):
	"""
	Extracts the following form a single signal:
	____________________________________________
	S.D. of raw amplitudes
	skew of raw amplitudes, mean ((X i -  Xmean)3
	mean of vertex-to-vertex a amplitudes
	S.D. of vertex-to-vertex amplitude
	coefficient of variation of vertex-to-vertex amplitude
	mean of absolute slopes of raw amplitudes, mean (abs(dx/dt))
	"""

	mean = signal.mean()
	amplitude = signal - mean
	SD_amplitude = amplitude.std()
	SKEW_amplitude = ((amplitude - mean) ** 3).mean()
	ind, vertices = create_vertex2vertex(signal)
	vertices = signal[ind]
	vertices_lag = shift(vertices, -1, cval=0)
	MEAN_v2v = (vertices - vertices_lag).mean()
	SD_v2v = (vertices - vertices_lag).std()
	CV_v2v = SD_v2v / MEAN_v2v

	slope_mean = abs(vertices[:-1] / vertices_lag[:-1]).mean()

	return SD_amplitude, SKEW_amplitude, MEAN_v2v, SD_v2v, CV_v2v, slope_mean
