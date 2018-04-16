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
	p = 50	# for generalized mean

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
			########
			## RMS
			small.append(RMS(X[j][i]))
			feature_names.append("rms_lead_" + str(j+1))
			########
			## harmonic
			small.append(harmonic(X[j][i]))
			feature_names.append("harmonic_lead_" + str(j+1))
			########
			## geometric
			small.append(geometric(X[j][i]))
			feature_names.append("geometric_lead_" + str(j+1))
			########
			## generalized
			small.append(generalized_mean(X[j][i], p))
			feature_names.append("generalized_lead_" + str(j+1))

			########
			## Piecewise Aggregate Approximation
			## (split series into parts and take mean of each0)
			## This makes sense as the neurons should be firing in aggregating f,)
			m1, m2, m3 = PAA(X)
			small.append(m1)
			feature_names.append("PAA1_" + str(j+1))
			small.append(m2)
			feature_names.append("PAA2_" + str(j+1))
			small.append(m3)
			feature_names.append("PAA3_" + str(j+1))

		all.append(small)
	all = np.asarray(all)
	return all, feature_names

def RMS (lead):
	sum = 0
	for i in range(len(lead)):
		sum += lead[i]**2
	return np.sqrt(sum/len(lead))

def harmonic (lead):
	sum = 0
	for i in range(len(lead)):
		sum += 1/lead[i]
	return len(lead)/sum

def geometric (lead):
	sum = 1
	for i in range(len(lead)):
		if lead[i] != 0:
			sum *= lead[i]
	return abs(sum)**(1/len(lead))

def generalized_mean (lead, p):
	sum = 1
	for i in range(len(lead)):
		sum *= lead[i]
	return abs(sum)**(1/p)

## Piecewise Aggregate Approximation
def PAA(X, split = 3):
	length = int(len(X)/split)
	m1 = X[:length]
	m2 = X[length:length+length]
	m3 = X[length+length:]

	m1 = m1.mean()
	m2 = m2.mean()
	m3 = m3.mean()
	return m1, m2, m3
