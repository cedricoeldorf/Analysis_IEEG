import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import variation
from scipy.ndimage.interpolation import shift
import peakutils, time


####################################
## Extract statistics for every lead and create AV table
####################################

def extract_basic(X):
	print("extracting basics")
	all = []  # will be the whole dataset
	p = 50  # for generalized mean
	# Iterate over every trial
	tic = time.time()
	tic_ = time.time()
	for trial in range(0, X.shape[1]):
		print(time.time() - tic)
		tic = time.time()
		small = np.array([])  # this is temporary list to add to new data set after every iteration
		feature_names = np.array([])  # for later feature extraction, we create a list of names

		# get every lead for current trial
		for lead in range(0, X.shape[0]):
			signal = X[lead][trial]
			peaks, valleys, signal_smooth = create_vertex2vertex(signal, spacing=10, seperate_peaks_valleys=True)
			vertices = np.append(peaks, valleys)
			vertices.sort()

			res = amplitude_features(signal, vertices, signal_smooth)
			AMSD = res[0]
			small = np.append(small, res)
			small = np.append(small, curvature_period_features(AMSD, vertices, signal_smooth, peaks, valleys))
			small = np.append(small, Amplitude(signal))
			small = np.append(small, crest(signal))
			small = np.append(small, RAPN(signal))
			small = np.append(small, RTRF(signal))
			small = np.append(small, RTPN(signal))
			small = np.append(small, RMS(signal))
			small = np.append(small, harmonic(signal))
			# small = np.append(small, geometric(signal))
			small = np.append(small, generalized_mean(signal, p))
			small = np.append(small, PAA(signal))
			small = np.append(small, absolute_slopes_features(vertices, signal_smooth))
			'''
            print(vertices) the positions in smoothed signal that were selected as vertices
            print(smoothed_signal[vertices]) the corresponding value for these positions

            ########
            ## mean
            small.append(signal.mean())
            feature_names.append("mean_lead_" + str(lead + 1))
            ########
            ## Max
            small.append(signal.max())
            feature_names.append("max_lead_" + str(lead + 1))
            ########
            ## Min
            small.append(signal.min())
            feature_names.append("min_lead_" + str(lead + 1))
            ########
            ## RMS
            small.append(RMS(signal))
            feature_names.append("rms_lead_" + str(lead + 1))
            ########
            ## harmonic
            small.append(harmonic(signal))
            feature_names.append("harmonic_lead_" + str(lead + 1))
            ########
            ## geometric
            small.append(geometric(signal))
            feature_names.append("geometric_lead_" + str(lead + 1))
            ########
            ## generalized
            small.append(generalized_mean(signal, p))
            feature_names.append("generalized_lead_" + str(lead + 1))

            ########
            ## Piecewise Aggregate Approximation
            ## (split series into parts and take mean of each0)
            ## This makes sense as the neurons should be firing in aggregating f,)
            m1, m2, m3 = PAA(X)
            small.append(m1)
            feature_names.append("PAA1_" + str(lead + 1))
            small.append(m2)
            feature_names.append("PAA2_" + str(lead + 1))
            small.append(m3)
            feature_names.append("PAA3_" + str(lead + 1))
            '''

	# all.append(small)
	all = np.asarray(all)
	return all, feature_names


def curvature_period_features(AMSD, vertices, signal_smooth, peaks, valleys):  # mean and S.D. coefficient of variation of curvatures at vertices
	"""
            mean of curvatures (d2x/dt2) at vertices
            S.D. of curvatures at vertices
            coefficient of variation of curvatures at vertices
            vertex counts/sec
            S.D. of vertex-to-vertex period
            coefficient of variation of vertex-to-vertex period
            count of mean crossings/sec (hysteresis = 25% of AMSD)
    """
	dx_dt = np.gradient(peaks)
	dy_dt = np.gradient(signal_smooth[peaks])
	d2x_dt2 = np.gradient(dx_dt)
	d2y_dt2 = np.gradient(dy_dt)
	peak_curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5

	dx_dt = np.gradient(valleys)
	dy_dt = np.gradient(signal_smooth[valleys])
	d2x_dt2 = np.gradient(dx_dt)
	d2y_dt2 = np.gradient(dy_dt)
	valley_curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5

	mean_curv_pos_over_mean_curv_neg = peak_curvature.mean() / valley_curvature.mean()

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

	return np.array([curvature.mean(),
					 curvature.std(),
					 variation(curvature),
					 vertices_per_second,
					 vertices_period.std(),
					 variation(vertices_period),
					 CTMXMN,
					 mean_curv_pos_over_mean_curv_neg])


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


def create_vertex2vertex(data, spacing=10, smooth=True, window=51, polyorder=3, seperate_peaks_valleys=False):
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
	if seperate_peaks_valleys:
		ind_valley.sort()
		ind_peak.sort()
		return ind_peak, ind_valley, data
	else:
		ind = np.append(ind_peak, ind_valley)
		ind.sort()
		return ind, data


# feature 21-26
def Amplitude(signal):
	para1 = round(0.5 * len(signal))  # to compute the amplitude 50% -100% frequency band
	para2 = round(0.25 * len(signal))  # 25% - 50%
	para3 = round(0.12 * len(signal))  # 12% - 25%
	para4 = round(0.06 * len(signal))  # 6% - 12%
	para5 = round(0.03 * len(signal))

	am1 = signal[para1:]
	Amplitude1 = (max(am1) - min(am1)) / 2
	am2 = signal[para2:para1]
	Amplitude2 = (max(am2) - min(am2)) / 2
	am3 = signal[para3:para2]
	Amplitude3 = (max(am3) - min(am3)) / 2
	am4 = signal[para4:para3]
	Amplitude4 = (max(am4) - min(am4)) / 2
	am5 = signal[para5:para4]
	Amplitude5 = (max(am5) - min(am5)) / 2
	am6 = signal[0:para5]
	Amplitude6 = (max(am6) - min(am6)) / 2
	return np.array([Amplitude1, Amplitude2, Amplitude3, Amplitude4, Amplitude5, Amplitude6])


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
	return np.array([max / rms])


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
	return np.array([mean_of_pos / mean_of_neg])


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
	return np.array([rise_mean / fall_mean])


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
	return np.array([mean_pos_crossing / mean_neg_crossing])


def RMS(lead):
	sum = 0
	for i in range(len(lead)):
		sum += lead[i] ** 2
	return np.array([np.sqrt(sum / len(lead))])


def harmonic(lead):
	sum = 0
	for i in range(len(lead)):
		sum += 1 / lead[i]
	return np.array([len(lead) / sum])


def geometric(lead):
	sum = 1
	for i in range(len(lead)):
		if lead[i] != 0:
			sum *= lead[i]
	return np.array([abs(sum) ** (1 / len(lead))])


def generalized_mean(lead, p):
	sum = 1
	for i in range(len(lead)):
		sum += lead[i]
	return np.array([(abs(sum) ** (1 / p)) / len(lead)])


#  Piecewise Aggregate Approximation
def PAA(X, split=3):
	length = int(len(X) / split)
	m1 = X[:length]
	m2 = X[length:length + length]
	m3 = X[length + length:]

	m1 = m1.mean()
	m2 = m2.mean()
	m3 = m3.mean()
	return np.array([m1, m2, m3])


def amplitude_features(signal, ind, vertices):
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
	vertices = vertices[ind]
	vertices_lag = shift(vertices, -1, cval=0)
	MEAN_v2v = (vertices - vertices_lag).mean()
	SD_v2v = (vertices - vertices_lag).std()
	CV_v2v = SD_v2v / MEAN_v2v

	slope_mean = abs(vertices[:-1] / vertices_lag[:-1]).mean()

	return np.array([SD_amplitude, SKEW_amplitude, MEAN_v2v, SD_v2v, CV_v2v, slope_mean])


def absolute_slopes_features(signal, ind, vertices):
	"""
Extracts the following form a single signal:
____________________________________________
7. S.D. of absolute slopes of raw amplitudes
8. coefficient of variation of absolute slopes of raw amplitudes
9. mean of vertex-to-vertex absolute slopes
10. S.D. of vertex-to-vertex absolute slopes
11. coefficient of variation of vertex-to-vertex absolute slopes
12. mean of curvatures (d2x/dt2) at vertices (already done by Rico????)
mean of absolute slopes of raw amplitudes, mean (abs(dx/dt)) - slope_MEAN
"""

	vertices = vertices[ind]
	vertices_lag = shift(vertices, -1, cval=1)
	mean = signal.mean()
	raw_amplitude = signal - mean
	# raw_amplitude = amplitude[ind]
	raw_amplitude_lag = shift(raw_amplitude, cval=0)
	# slope_amplitude = abs(raw_amplitude[:-1] / raw_amplitude_lag[:-1])
	slope_amplitude = abs(raw_amplitude / raw_amplitude_lag)
	slope_MEAN = slope_amplitude.mean()
	slope_SD = slope_amplitude.std()
	CV_slope_amplitude = slope_SD / slope_MEAN
	slope_v2v = abs(vertices[:-1] / vertices_lag[:-1])
	MEAN_v2v_slope = slope_v2v.mean()
	SD_v2v_slope = slope_v2v.std()
	CV_v2v_slope = SD_v2v_slope / MEAN_v2v_slope

	return np.array([slope_SD, CV_slope_amplitude, MEAN_v2v_slope, SD_v2v_slope, CV_v2v_slope])
