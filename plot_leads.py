import numpy as np
import pickle
import matplotlib.pyplot as plt

num_of_leads_to_plot = 2

with open ('C:\FILES\MAASTRICHT UNIVERSITY\RESEARCH PROJECT 2\X_memory.pkl', 'rb') as fp:
    X_memory = pickle.load(fp)
with open ('C:\FILES\MAASTRICHT UNIVERSITY\RESEARCH PROJECT 2\y_memory.pkl', 'rb') as fp:
    y_memory = pickle.load(fp)
with open ('C:\FILES\MAASTRICHT UNIVERSITY\RESEARCH PROJECT 2\X_perc.pkl', 'rb') as fp:
    X_perc = pickle.load(fp)
with open ('C:\FILES\MAASTRICHT UNIVERSITY\RESEARCH PROJECT 2\y_perc.pkl', 'rb') as fp:
    y_perc = pickle.load(fp)

for i in range(0, num_of_leads_to_plot):
    plt.plot(X_memory[i])

plt.show()
