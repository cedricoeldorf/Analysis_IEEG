import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import math
from scipy.fftpack import fft, irfft, rfft
from scipy.optimize import curve_fit
import tsfresh.feature_extraction.feature_calculators as ts

from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
from tsfresh import extract_features
download_robot_execution_failures()
timeseries, y = load_robot_execution_failures()


def normalize(signal, range=None, offset=None):

    '''
    # If range = None and Offset = None:
        - return normalized signal with values in range (0,1)
    # Range squeezes the signal between range(-range, +range)
    # Offet adds offset...
    '''
    norm_sig = (signal - np.min(signal))/(np.max(signal) - np.min(signal))
    if range is not None:
        norm_sig = (2*norm_sig - 1)*range
    if offset is not None:
        norm_sig = norm_sig + offset
    return norm_sig

# with open('/Users/jiayun/PycharmProjects/D'
#           'ecision making projext/partner/Analysis_IEEG/ZHAO/f'
#           'iltered_lead_1_mem.pkl', 'rb') as fp:
#     x= pickle.load(fp)
# with open('/Users/jiayun/PycharmProjects/Decision making projext/p'
#           'artner/Analysis_IEEG/ZHAO/y.pkl', 'rb') as fp:
#     y = pickle.load(fp)
# # extracted_features = extract_features(timeseries, column_id = 'id', column_sort = 'time')
#
# kurtosis = ts.kurtosis(signal)
# k_l = []
# for i in range(x.shape[0]-1):
#
#     signal = x[i, :]
#     kurto_i = ts.kurtosis(signal)
#     kurto_i = round(kurto_i,3)
#     k_l.append(kurto_i)
#     k_l = np.asarray(k_l)
#     # k = np.asarray(k_l)
# print (k_l)
#########################-------------------#########################
#               now test other leads
#########################-------------------#########################

with open('/Users/jiayun/PycharmProjects/D'
          'ecision making projext/partner/Analysis_IEEG/ZHAO/'
          'eeg_m_filtered.pkl', 'rb') as fp_m:
    memory_data = pickle.load(fp_m)

with open('/Users/jiayun/PycharmProjects/D'
          'ecision making projext/partner/Analysis_IEEG/ZHAO/e'
          'eg_p_filtered.pkl', 'rb') as fp_p:
    perception_data = pickle.load(fp_p)

kurto_matrix = []
for i in range(memory_data.shape[0]):
    l1 = memory_data[i]
    k_l = []
    for j in range(l1.shape[0]):
        signal = l1[j, :]
        signal = normalize(signal, range=3, offset=None)
        kurto_j = ts.kurtosis(signal)
        kurto_j = round(kurto_j,3)
        k_l.append(kurto_j)
        k_j = np.asarray(k_l)
    kurto_matrix.append(k_j)

k_m = np.asarray(kurto_matrix)
unavaliable_experments_m = np.where(k_m >= 15)
unavaliable_experments_m = np.asanyarray(unavaliable_experments_m)
print("Unavaliable memory experiments dete"
      "cted, '[lead], [trail]':", unavaliable_experments_m[0] , unavaliable_experments_m[1])



kurto_matrix = []
for i in range(perception_data.shape[0]):
    l1 = perception_data[i]
    k_l = []
    for j in range(l1.shape[0]):
        signal = l1[j, :]
        signal = normalize(signal, range=3, offset=None)
        kurto_j = ts.kurtosis(signal)
        kurto_j = round(kurto_j,3)
        k_l.append(kurto_j)
        k_j = np.asarray(k_l)
    kurto_matrix.append(k_j)

k_p = np.asarray(kurto_matrix)
unavaliable_experments_p = np.where(k_p >= 15)
unavaliable_experments_p = np.asanyarray(unavaliable_experments_p)
print("Unavaliable perception experimen"
      "ts detected, '(lead), (trail)':", unavaliable_experments_p[0], unavaliable_experments_p[1])
