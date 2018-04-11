import numpy as np
import pickle

# hi all

with open('preprocessed/X_memory.pkl', 'rb') as fp:
	X_memory = pickle.load(fp)

with open('preprocessed/X_perc.pkl', 'rb') as fp:
	X_perc = pickle.load(fp)

trials = X_memory.shape[0]
leads = X_memory.shape[1]
time_steps = X_memory.shape[2]

total = [[] for i in range(leads)]
print(total)
for lead in range(0,leads):
	for trial in range(0,trials):
		entry = X_memory[trial][lead]
		total[lead].append(entry)

for lead in range(leads):
	print(total[lead][0])


def extract_basic():
	print("extracting basics")


def extract_advanced():
	print("extracting advanced")
