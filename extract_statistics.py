import numpy as np
import pickle
import pandas as pd

######################################
### Read in files
######################################

def extract_leads():
	with open('preprocessed/X_memory.pkl', 'rb') as fp:
		X_memory = pickle.load(fp)

	with open('preprocessed/X_perc.pkl', 'rb') as fp:
		X_perc = pickle.load(fp)

	####################################
	## Extract seperate leads
	###################################

	trials = X_memory.shape[0]
	leads = X_memory.shape[1]
	time_steps = X_memory.shape[2]

	total_mem = [[] for _ in range(leads)]
	for lead in range(0, leads):
		for trial in range(0, trials):
			entry = X_memory[trial][lead]
			total_mem[lead].append(entry)

	trials = X_perc.shape[0]
	leads = X_perc.shape[1]
	time_steps = X_perc.shape[2]

	total_perc = [[] for _ in range(leads)]
	for lead in range(0, leads):
		for trial in range(0, trials):
			entry = X_perc[trial][lead]
			total_perc[lead].append(entry)

	return np.asarray(total_mem), np.asarray(total_perc)


####################################
## Extract statistics for every lead and create AV table
####################################

def extract_basic(X):
	print("extracting basics")

	all = []  # will be the whole dataset

	# Iterate over every trial
	for i in range(0, X.shape[1]):
		small = []  # this is temporary list to add to new data set after every iteration
		feature_names = []  # for later feature extraction, we create a list of names

		# get every lead for current trial
		for j in range(0, X.shape[0]):
			########
			## mean
			small.append(X[j][i].mean())
			feature_names.append("mean_lead_" + str(j + 1))
			########
			## Max
			small.append(X[j][i].max())
			feature_names.append("max_lead_" + str(j + 1))
			########
			## Min
			small.append(X[j][i].min())
			feature_names.append("min_lead_" + str(j + 1))

		all.append(small)
	all = np.asarray(all)
	return all, feature_names


def extract_advanced(X):
	print("extracting advanced")
