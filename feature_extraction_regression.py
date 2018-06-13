#################################
## Measuring Feature importance by slope/target correlation
##################################
from extract_statistics import *
from sklearn.metrics import accuracy_score
import numpy as np
from scipy import stats
from copy import deepcopy
from multiprocessing import Pool, Manager, cpu_count, Queue
import pickle
from load_raw import load_raw
target = [0,1,0,1,0,1,1,0,1,0]
feature_names = ['f1','f2','f3']
#############3
# We have a lead. All trials, in each trial split into time segments
# first: get features of every time segment: for trial, for segment, append
# get: a list of features in a list of segments for each trial


## This is one time segmented array of features (one lead)
## Run this for a specific lead over all trials
## Start with array that has n features in k segments


def get_slopes(data, feature_names):
    #for every lead
    data = np.nan_to_num(data)
    data = data.reshape(data.shape[0],data.shape[1],data.shape[3],data.shape[2])
    slopes = []
    # LEAD
    for i in range(data.shape[0]):
        # for every trial
        # TRIAL
        trial_slopes = []
        for j in range(data.shape[1]):
            # 3 faetures in 9 segments

            n_features = data.shape[3]

            ## FEATURE
            slope = []
            for k in range(data.shape[2]):

                ## FEATURE
                f = data[i][j][k].ravel()
                x = list(range(n_features))
                m,b = np.polyfit(x, f, 1)
                slope.append(m)
            trial_slopes.append(slope)

        #print(trial_slopes[0])
        slopes.append(trial_slopes)
    return slopes

def average_slope(slopes, feature_names, removal = False, bottom_thresh = 0, top_thresh = 0):
    # INCOMPLETE
    mean_slopes = []

    # LEAD
    for z in range(len(slopes)):

        # take lead, transpose in order to compare every target
        slopes[z] = np.asarray(slopes[z]).T
        sl = []
        for x in range(len(slopes[z])):
            av_sl = slopes[z][x].mean()
            sl.append(av_sl)
        mean_slopes.append(sl)

    if removal == True:
        final_slopes = deepcopy(mean_slopes)
        temp_slopes = deepcopy(mean_slopes)
        temp_slopes = np.asarray(temp_slopes).T
        for i in range(len(temp_slopes)):
            print("MIN: " + str(temp_slopes[i].min()))
            print("MAX: " + str(temp_slopes[i].max()))
            index = []
            if (temp_slopes[i].min() > bottom_thresh) and (temp_slopes[i].max() < top_thresh):

                index.append(i)
        final_slopes = np.delete(final_slopes, index, 0)
        final_slopes = np.delete(feature_names, index, 0)
        mean_slopes = deepcopy(final_slopes)

    return mean_slopes, feature_names

def get_importances(slopes,target, feature_names):

    #############################
    ## Get coherence with target
    #############################
    for i in range(len(slopes)):
        for j in range(len(slopes[i])):
            for x in range(len(slopes[i][j])):
                if slopes[i][j][x] < 0:
                    slopes[i][j][x] = 0
                else:
                    slopes[i][j][x] = 1


    importance = []
    # iterate over leads
    for z in range(len(slopes)):
        # take lead, transpose in order to compare every target
        slopes[z] = np.asarray(slopes[z]).T
        # now, for every feature, get accuracy
        acc = []
        for x in range(len(slopes[z])):
            acc.append(accuracy_score(target,slopes[z][x]))

        #print(len(acc))
        #print(importance)
        importance.append(acc)
    return importance

with open('./preprocessed/eeg_split/bin_mem.pkl', 'rb') as f:
	eeg_m =	pickle.load(f)

feature_names = eeg_m[1]
eeg_m = eeg_m[0]
patient_data = load_raw('raw_FAC002')
target = patient_data['simVecM'][0:eeg_m.shape[1]]
del patient_data

slopes = get_slopes(eeg_m, feature_names)
s = deepcopy(slopes)
mean_slope, feature_names = average_slope(s, feature_names, removal = True, bottom_thresh = -0.1, top_thresh = 0.1)
print(feature_names)
s = deepcopy(slopes)
important = get_importances(s,target, feature_names)


'''######################################
## attempt to multithread get_slopes function '''

def extract_multithreaded_basic(X):
    pool = Pool(cpu_count())

    m = Manager()
    queue = m.Queue()  # create queue to save all the results

    tasks = []
    for lead in range(X.shape[0]):
        for trial in range(X.shape[1]):
            for segment in range(X.shape[2]):
                for feature in range(X.shape[3]):
                    n_feat = X.shape[3]
                    feat = segment.T[feature]
                    tasks.append([feat, n_feat, queue])
    print(len(tasks))
    pool.map(execute, tasks)  # create the results
    return queue


def execute(args):
    f, n_feat, queue = args[0], args[1], args[2]
    x = list(range(0,n_feat))
    m,b = np.polyfit(x, f, 1)
    queue.put((m, b))
