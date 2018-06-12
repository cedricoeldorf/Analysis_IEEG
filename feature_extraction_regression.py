#################################
## Measuring Feature importance by slope/target correlation
##################################
from extract_statistics import *
from sklearn.metrics import accuracy_score
import numpy as np
from scipy import stats
target = [0,1,0,1,0,1,1,0,1,0]
#############3
# We have a lead. All trials, in each trial split into time segments
# first: get features of every time segment: for trial, for segment, append
# get: a list of features in a list of segments for each trial


## This is one time segmented array of features (one lead)
## Run this for a specific lead over all trials
## Start with array that has n features in k segments



#for every lead
feature_names = ['f1','f2','f3']
leads = []
leads_importances = []
for i in range(125):
    # for every trial
    trial_slopes = []
    for i in range(10):
        # 3 faetures in 9 segments
        features = np.asarray([[np.random.uniform(-1,1, size = 3)],[np.random.uniform(-1,1, size = 3)],[np.random.uniform(-1,1, size = 3)],[np.random.uniform(-1,1, size = 3)],[np.random.uniform(-1,1, size = 3)],[np.random.uniform(-1,1, size = 3)],[np.random.uniform(-1,1, size = 3)],[np.random.uniform(-1,1, size = 3)],[np.random.uniform(-1,1, size = 3)]])
        features = features.T
        n_features = len(features)

        slopes = []
        for i in range(n_features):
            f = features[i].ravel()
            x = list(range(len(f)))
            m,b = np.polyfit(x, f, 1)
            slopes.append(m)
        trial_slopes.append(slopes)

    leads.append(trial_slopes)

    #############################
    ## Get coherence with target
    #############################
    for z in range(len(trial_slopes)):
        for x in range(len(trial_slopes[z])):
            if trial_slopes[z][x] < 0:
                trial_slopes[z][x] = 0
            else:
                trial_slopes[z][x] = 1
    trial_slopes = np.asarray(trial_slopes).T

    importance = []
    for z in range(len(trial_slopes)):
        importance.append(accuracy_score(target, trial_slopes[z]))
    leads_importances.append(importance)





# trial_slopes = np.asarray(trial_slopes).T
#
# importance  = []
# for i in range(len(trial_slopes)):
#
