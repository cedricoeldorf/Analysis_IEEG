## BRAIN VIS
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
path = './preprocessed/lead_coordinates/patient_2/'

coordinates = pd.read_csv(path + 'coords.txt', sep='\t')
chosen = pd.read_csv(path + 'good_leads.txt', sep='\t')
chosen = chosen['lead'].tolist()

coordinates = coordinates.set_index(coordinates.lead)
coordinates = coordinates.drop('lead', axis = 1)
coordinates = coordinates.ix[chosen]
coordinates = coordinates.reset_index()

for i in range(len(coordinates)):
    coordinates.coord[i] = [x.strip() for x in coordinates.coord[i].split(',')]

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(int(91), int(127), int(73), c='b')
# for p in coordinates.coord:
#     print(p)
#     ax.scatter(int(p[0]), int(p[1]), int(p[2]), c='r')
# plt.ion()
#plt.show()


## accuracies
filename = 'accuracies_raw_FAC002_True_880_False_220_False_theta_alpha_True.pkl'
with open('./preprocessed/accuracies/' + filename, 'rb') as f:
    ac = pickle.load(f)

plotting = []
for i in range(0, len(ac[4])):
    plotting.append(ac[4][i][0])

coordinates.area = coordinates.area.str.replace('\d+', '')

coordinates['acc'] = plotting

plot = pd.DataFrame()
plot['acc'] = coordinates.groupby(['area'])['acc'].mean()
plot['std'] = coordinates.groupby(['area'])['acc'].std()
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

plot = plot.sort_values(by=['acc'],ascending=False)

plot['acc'].plot.bar(color=tableau20, yerr = plot['std'])
plt.title('Predicition Accuracy for each Brain Region')
plt.xlabel("Region")
plt.ylabel("Accuracy")
plt.ion()
plt.tight_layout()
plt.show()
