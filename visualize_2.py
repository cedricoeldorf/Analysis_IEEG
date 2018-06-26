import numpy as np
import seaborn as sb
import random
import matplotlib.pyplot as plt
import math
import pylab
from load_pickle import load_pickle, save_pickle
## BRAIN VIS
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
import glob

''' ############################################################################
## Load files
########################################################################### '''


def load_data(path):
    path_importances = path + '*.pkl'
    files_importances = glob.glob(path_importances)
    files_importances = [files_importances[file] for file in range(len(files_importances))]

    patient_ids = []
    task_type = []
    for i in range(len(files_importances)):
        start_loc_id = files_importances[i].find('FAC')
        start_loc_type = files_importances[i].find('features_')
        patient_ids.append(files_importances[i][start_loc_id+3:start_loc_id+6])
        task_type.append(files_importances[i][start_loc_type+len('features_'):start_loc_type+len('features_')+1])

    patient_ids = list(set(patient_ids))
    patient_data = []
    patient_name_list = []

    for i in range(len(files_importances)):
            data = load_pickle(files_importances[i])
            for j in range(len(patient_ids)):
                if patient_ids[j] in files_importances[i]:
                    patient_data.append(data)
                    if task_type[i] == 'm':
                        type = 'Memory'
                    else:
                        type = 'Perception'
                    patient_name_list.append(int(patient_ids[j]))
                    break

    return patient_data, patient_name_list


''' ###########################################################################
## Plot SLOPES
########################################################################### '''


def plot_feature_regression(m, feature_name, b=0, n_windows=2.2, threshold=1, targets=None, save=False):

    if isinstance(m, float):
        fig = plt.figure()
        x = np.linspace(0, n_windows, endpoint=True)
        y = m*x + b
        ax = fig.add_subplot(1, 1, 1)
        if targets is not None:
            if targets[i] == 1.0:
                notation = '+'
            else:
                notation = '-'
        ax.set_ylim(-threshold, threshold)
        ax.plot(x, y)
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('center')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_smart_bounds(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.title('Correlation over time of ' + feature_name)
        plt.xlabel('Time')
        plt.ylabel('Correlation')
        plt.show()
    else:
        fig = plt.figure(figsize=(30.0, 12.0))
        plt.suptitle('Lead #13, Feature #41')
        n_rows, n_cols = optimize_grid(len(m))
        ax = [pylab.subplot(n_rows, n_cols, v) for v in range(1, len(m)+1)]
        for i in range(len(m)):
            x = np.linspace(0, n_windows, endpoint=True)
            y = m[i]*x + b
            if targets is not None:
                if targets[i] == 1.0:
                    notation = '+'
                else:
                    notation = '-'
                ax[i].annotate(notation, xy=(0.5,0.5), )
            ax[i].set_ylim(-threshold, threshold)
            ax[i].plot(x, y)
            ax[i].spines['left'].set_position('zero')
            ax[i].spines['right'].set_color('none')
            ax[i].spines['bottom'].set_position('center')
            ax[i].spines['top'].set_color('none')
            ax[i].spines['bottom'].set_smart_bounds(True)
            ax[i].xaxis.set_ticks_position('bottom')
            ax[i].yaxis.set_ticks_position('left')
            ax[i].set_xlabel('Time')
            ax[i].set_ylabel('Correlation')
            if i <=1:
                ax[i].annotate('"YES"', fontsize=15, xy=(0.1,0.45))
            else:
                ax[i].annotate('"NO"', fontsize=15, xy=(0.1,0.45))
        if not save:
            plt.show()

    if save:
        plt.savefig('slopes/' + 'L_' + str(feature_name[0]) + '_F_ ' +str(feature_name[1]), dpi = 100)


''' ############################################################################
## Optimize grid

## Is prime
########################################################################### '''


def optimize_grid(num):

    if num == 1 or num == 2 or num == 3:
        row = 1
        col = num
        return row, col

    list = []
    for i in range(1, int(np.floor(np.sqrt(num))+1)):
        for j in range(1, num+1):
            if i*j==num:
                list.append([i, j])
    # print(list)
    row = list[-1][0]
    col = list[-1][1]
    return row, col


def isprime(num):
    for i in range(2, int(np.floor(np.sqrt(num)))+1):
        if num % i == 0:
            return False
    return True


''' ############################################################################
## Plot feature importance
########################################################################### '''


def plot_feature_importance(X, feature_names, filter=True, LOW=0.42, HIGH=0.58, show=True, patient_num=None, decision_type=None, brain_region_list=None):

    n_features = X.shape[1]
    n_leads = X.shape[0]
    n_rows, n_cols = optimize_grid(n_features)
    ax = [pylab.subplot(n_rows, n_cols, v) for v in range(1, n_features+1)]
    for i in range(n_features):
        ax[i].set_title(feature_names[i])
        ax[i].spines['top'].set_color('none')
        ax[i].spines['right'].set_color('none')
        ax[i].set_xlabel('Region')
        ax[i].set_ylabel('Accuracy')
        if (i+1) % n_cols != 0:
            ax[i+1].spines['left'].set_color('none')
            ax[i+1].axes.get_yaxis().set_visible(False)
        ax[i].set_ylim(LOW-0.05, HIGH+0.05)
        for j in range(n_leads):
            '''if filter:
                if X[j][i] > LOW and X[j][i] < HIGH:
                    X[j][i] = 0
            '''
            if X[j][i] > HIGH:
                color = 'g'
            elif X[j][i] < LOW:
                color = 'r'
            else:
                color = '#585e58'
            if brain_region_list is not None:
                x = np.array(np.arange(len(brain_region_list)))
                ax[i].set_xticks(x)
                ax[i].set_xticklabels(brain_region_list, rotation=45)
                for tick in ax[i].xaxis.get_major_ticks():
                    tick.label.set_fontsize(10)
            ax[i].bar(j, X[j][i], color=color)
            # for y in range(int(100*LOW), int(100*HIGH), 1):
            #     ax[i].plot(range(0,9), [y/100] * len(range(0,9)), "--", lw=0.5, color="black", alpha=0.3)
            ax[i].axhline(LOW, linewidth=1)
            ax[i].axhline(HIGH, linewidth=1)
    plt.tight_layout()

    if patient_num is not None and decision_type is not None:
        plt.suptitle('Feature Importances for Patient ' + patient_num + ' for ' + decision_type + ' Decision')
    if show:
        plt.show()


''' ############################################################################
## Find expressive lead+feature pairs
########################################################################### '''


def find_expressive_pairs(X, feature_names, LOW=0.1, HIGH=0.9):
    n_features = X.shape[1]
    n_leads = X.shape[0]
    pair_list = []
    for i in range(n_leads):
        for j in range(n_features):
            if X[i][j] <= LOW or X[i][j] >= HIGH:
                pair_list.append(['L:', i, 'F:', j, 'val:', X[i][j], 'name:', feature_names[j]])
    return pair_list


''' ############################################################################
## Features to keep
########################################################################### '''


def features_to_keep(important, LOW=0.42, HIGH=0.58):
    keepers_all = []
    for lead in range(len(important)):
    	keepers_feature = []
    	for feature in range(len(important[lead])):
    		if (important[lead][feature] < LOW) or  (important[lead][feature] > HIGH):
    			keepers_feature.append(feature)
    	keepers_all.append(keepers_feature)
    return keepers_all


''' ############################################################################
## Break plots
########################################################################### '''


def break_plots(data, names, split=6):

    print('Num of features:',data.shape[1])

    if data.shape[1] != len(names):
        print('Something is wrong...')
        return
    list_data = []
    list_names =  []
    pos = 0
    for i in range(data.shape[1]):
        if pos + split > data.shape[1]:
            if isprime(data.shape[1] - pos) and (data.shape[1] - pos) != 2:
                if pos != data.shape[1] - 1:
                    list_data.append(data[:,pos:-1])
                    list_names.append(names[pos:-1])
                else:
                    list_data.append(data[:,-1:])
                    list_names.append(names[-1])
            else:
                list_data.append(data[:,pos:])
                list_names.append(names[pos:])
            break
        else:
            list_data.append(data[:,pos:pos+split])
            list_names.append(names[pos:pos+split])
        pos += split
    return list_data, list_names


''' ############################################################################
## Plot importances per brain region
########################################################################### '''


def plot_brain_region(path_importances, path_brain_regions, set_threshold=True, plot=True):

    patient_substr = 'patient_'
    mem_substr = 'mem'
    patient_num = path_brain_regions[path_brain_regions.find(patient_substr) + len(patient_substr)]
    decision_type = 'Perception'
    if mem_substr in path_importances:
        decision_type = 'Memory'

    # importances = np.asarray(load_pickle(path_importances))
    importances = np.asarray(path_importances)
    if set_threshold:
        HIGH = 1.05*0.5 * (np.mean(importances) + np.max(importances))
        LOW = 0.95*0.5 * (np.mean(importances) + np.min(importances))
    else:
        HIGH = 0.6
        LOW = 0.4

    print('LOW', LOW)
    print('HIGH', HIGH)

    n_leads = importances.shape[0]
    n_features = importances.shape[1]

    good_features = features_to_keep(importances, LOW=LOW, HIGH=HIGH)

    list_of_features = []
    for i in range(len(good_features)):
        list_of_features += good_features[i]

    feature_indices = sorted(list(set(list_of_features)))

    importances_filtered = np.zeros((n_leads, len(feature_indices)))

    j=0
    for i in range(len(feature_indices)):
        importances_filtered[:,j] = importances[:,feature_indices[i]]
        j+=1

    feature_names = ['F' + str(feature_indices[i]) for i in range(len(feature_indices))]

    coordinates = pd.read_csv(path_brain_regions + 'coords.txt', sep='\t', error_bad_lines=False)
    chosen = pd.read_csv(path_brain_regions + 'good_leads.txt', sep='\t', error_bad_lines=False)
    chosen = chosen['lead'].tolist()
    coordinates = coordinates.set_index(coordinates.lead)
    coordinates = coordinates.drop('lead', axis=1)
    coordinates = coordinates.ix[chosen]
    coordinates = coordinates.reset_index()
    for i in range(len(coordinates)):
        coordinates.coord[i] = [x.strip() for x in coordinates.coord[i].split(',')]
    coordinates.area = coordinates.area.str.replace('\d+', '')

    prev = coordinates['area'][0]
    brain_region_list = [prev]
    l = []
    areas = []
    for i in range(len(coordinates)):
        if prev != coordinates['area'][i]:
            areas.append(l)
            l = []
            l.append(i)
            brain_region_list.append(coordinates['area'][i])
        else:
            l.append(i)
        prev = coordinates['area'][i]
        if i == len(coordinates) - 1:
            areas.append(l)

    brain_region = np.zeros((len(areas), len(feature_indices)))
    for i in range(len(areas)):
        for k in range(len(feature_indices)):
            extmax = np.max(importances_filtered[areas[i][0]:areas[i][-1] + 1, k])
            extmin = np.min(importances_filtered[areas[i][0]:areas[i][-1] + 1, k])
            if extmax > HIGH and extmin > LOW:
                brain_region[i][k] = extmax
            elif extmin < LOW and extmax < HIGH:
                brain_region[i][k] = extmin
            else:
                brain_region[i][k] = np.mean(importances_filtered[areas[i][0]:areas[i][-1] + 1, k])
    if plot:
        n_plots = brain_region.shape[1]
        brain_list, name_list = break_plots(brain_region, feature_names)
        for i in range(len(brain_list)):
            plot_feature_importance(brain_list[i], name_list[i], filter=True, LOW=LOW, HIGH=HIGH, show=True, patient_num=patient_num, decision_type=decision_type, brain_region_list=brain_region_list)

    # return importances_filtered, feature_indices, feature_names, brain_region, good_features, areas, coordinates, chosen, HIGH, LOW, brain_region_list


''' ############################################################################
## EX
########################################################################### '''

if __name__ == '__main__':
    
    path_importances = 'preprocessed/importances/'
    path_brain_regions = './preprocessed/lead_coordinates/'

    patient_data, patient_ids = load_data(path_importances)
    brain_regions = [path_brain_regions + 'patient_' + str(patient_ids[i]) + '/' for i in range(len(patient_ids))]

    FROM = 3
    TO = len(patient_ids)
    for i in range(FROM, TO):
        plot_brain_region(patient_data[i], brain_regions[i], set_threshold=True)
