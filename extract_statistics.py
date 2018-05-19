import numpy as np
from scipy.signal import savgol_filter


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

			# ########
			# ## mean
			# small.append(signal.mean())
			# feature_names.append("mean_lead_" + str(j + 1))
			# ########
			# ## Max
			# small.append(signal.max())
			# feature_names.append("max_lead_" + str(j + 1))
			# ########
			# ## Min
			# small.append(signal.min())
			# feature_names.append("min_lead_" + str(j + 1))
			# ########
			# ## RMS
			# small.append(RMS(signal))
			# feature_names.append("rms_lead_" + str(j + 1))
			# ########
			# ## harmonic
			# small.append(harmonic(signal))
			# feature_names.append("harmonic_lead_" + str(j + 1))
			# ########
			# ## geometric
			# small.append(geometric(signal))
			# feature_names.append("geometric_lead_" + str(j + 1))
			# ########
			# ## generalized
			# small.append(generalized_mean(signal, p))
			# feature_names.append("generalized_lead_" + str(j + 1))
			#
			# ########
			# ## Piecewise Aggregate Approximation
			# ## (split series into parts and take mean of each0)
			# ## This makes sense as the neurons should be firing in aggregating f,)
			# m1, m2, m3 = PAA(X)
			# small.append(m1)
			# feature_names.append("PAA1_" + str(j + 1))
			# small.append(m2)
			# feature_names.append("PAA2_" + str(j + 1))
			# small.append(m3)
			# feature_names.append("PAA3_" + str(j + 1))

		# all.append(small)
	all = np.asarray(all)
	return all, feature_names


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
	return list(ind), data


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


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
	r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
	The Savitzky-Golay filter removes high frequency noise from data.
	It has the advantage of preserving the original shape and
	features of the signal better than other types of filtering
	approaches, such as moving averages techniques.
	Parameters
	----------
	y : array_like, shape (N,)
		the values of the time history of the signal.
	window_size : int
		the length of the window. Must be an odd integer number.
	order : int
		the order of the polynomial used in the filtering.
		Must be less then `window_size` - 1.
	deriv: int
		the order of the derivative to compute (default = 0 means only smoothing)
	Returns
	-------
	ys : ndarray, shape (N)
		the smoothed signal (or it's n-th derivative).
	Notes
	-----
	The Savitzky-Golay is a type of low-pass filter, particularly
	suited for smoothing noisy data. The main idea behind this
	approach is to make for each point a least-square fit with a
	polynomial of high order over a odd-sized window centered at
	the point.
	Examples
	--------
	t = np.linspace(-4, 4, 500)
	y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
	ysg = savitzky_golay(y, window_size=31, order=4)
	import matplotlib.pyplot as plt
	plt.plot(t, y, label='Noisy signal')
	plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
	plt.plot(t, ysg, 'r', label='Filtered signal')
	plt.legend()
	plt.show()
	References
	----------
	.. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
	   Data by Simplified Least Squares Procedures. Analytical
	   Chemistry, 1964, 36 (8), pp 1627-1639.
	.. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
	   W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
	   Cambridge University Press ISBN-13: 9780521880688
	"""
	import numpy as np
	from math import factorial

	try:
		window_size = np.abs(np.int(window_size))
		order = np.abs(np.int(order))
	except ValueError as msg:
		raise ValueError("window_size and order have to be of type int")
	if window_size % 2 != 1 or window_size < 1:
		raise TypeError("window_size size must be a positive odd number")
	if window_size < order + 2:
		raise TypeError("window_size is too small for the polynomials order")
	order_range = range(order + 1)
	half_window = (window_size - 1) // 2
	# precompute coefficients
	b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
	m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
	# pad the signal at the extremes with
	# values taken from the signal itself
	firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
	lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
	return np.convolve(m[::-1], y, mode='valid')
