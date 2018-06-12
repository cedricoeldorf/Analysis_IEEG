import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pylab

''' ############################################################################
## Plot feature correlation
########################################################################### '''

def plot_feature_regression(m, b, n_windows, threshold, feature_name):

    fig = plt.figure()
    x = np.linspace(0, n_windows, endpoint=True)
    y = m*x + b
    ax = fig.add_subplot(1, 1, 1)
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

''' ############################################################################
## Plot feature importance
########################################################################### '''

def optimize_grid(num):
    list = []
    for i in range(1, int(np.floor(np.sqrt(num))+1)):
        for j in range(1, num+1):
            if i*j==num:
                list.append([i, j])
    return list[-1][0], list[-1][1]

def plot_feature_importance(X, feature_names):
    n_features = X.shape[1]
    n_leads = X.shape[0]
    n_rows, n_cols = optimize_grid(n_features)
    ax = [pylab.subplot(n_rows, n_cols, v) for v in range(1, n_features+1)]
    for i in range(n_features):
        for j in range(n_leads):
            ax[i].bar(j, X[j][i])
            ax[i].set_title(feature_names[i])
            ax[i].spines['top'].set_color('none')
            ax[i].spines['right'].set_color('none')
            if (i+1) % n_cols != 0:
                ax[i+1].spines['left'].set_color('none')
                ax[i+1].axes.get_yaxis().set_visible(False)
    plt.show()

''' ############################################################################
## EX
########################################################################### '''

m = 0.1
b = 0
n_windows = 10
threshold = 1

n_feat = 15

labels = ['feature' + str(i) for i in range(1, n_feat+1)]

A = np.asarray([[np.random.randint(10)/10 for _ in range(n_feat)] for _ in range(20)])

plot_feature_importance(A, labels)
plot_feature_regression(m, b, n_windows, threshold, 'puffpaff')
