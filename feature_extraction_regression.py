#################################
## Measuring Feature importance by slope/target correlation
##################################
from extract_statistics import *
from sklearn.metrics import accuracy_score
import numpy as np
from scipy import stats
from copy import deepcopy
target = [0,1,0,1,0,1,1,0,1,0]
feature_names = ['f1','f2','f3']
#############3
# We have a lead. All trials, in each trial split into time segments
# first: get features of every time segment: for trial, for segment, append
# get: a list of features in a list of segments for each trial


## This is one time segmented array of features (one lead)
## Run this for a specific lead over all trials
## Start with array that has n features in k segments


def get_slopes(feature_names):
    #for every lead

    slopes = []
    for i in range(125):
        # for every trial
        trial_slopes = []
        for j in range(10):
            # 3 faetures in 9 segments
            features = np.asarray([[np.random.uniform(-1,1, size = 3)],[np.random.uniform(-1,1, size = 3)],[np.random.uniform(-1,1, size = 3)],[np.random.uniform(-1,1, size = 3)],[np.random.uniform(-1,1, size = 3)],[np.random.uniform(-1,1, size = 3)],[np.random.uniform(-1,1, size = 3)],[np.random.uniform(-1,1, size = 3)],[np.random.uniform(-1,1, size = 3)]])
            features = features.T
            n_features = len(features)

            slope = []
            for k in range(n_features):
                f = features[k].ravel()
                x = list(range(len(f)))
                m,b = np.polyfit(x, f, 1)
                slope.append(m)
            trial_slopes.append(slope)

        #print(trial_slopes[0])
        slopes.append(trial_slopes)
    return slopes

def average_slope(slopes, feature_names, removal = False, bottom_thresh = 0, top_thresh = 0):
    # INCOMPLETE
    mean_slopes = []
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
                final_slopes = np.delete(final_slopes, i, 0)
                index.append(i)
        del feature_names[i]
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

slopes = get_slopes(feature_names)
s = deepcopy(slopes)
mean_slope, feature_names = average_slope(s, feature_names, removal = True, bottom_thresh = -0.1, top_thresh = 0.1)
print(feature_names)
s = deepcopy(slopes)
important = get_importances(s,target, feature_names)
