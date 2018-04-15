import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from extract_statistics import extract_leads, extract_basic
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

np.random.seed(0)

X_mem, X_perc = extract_leads()
y_mem = pickle.load(open('preprocessed/y_memory.pkl', 'rb'))
y_perc = pickle.load(open('preprocessed/y_perc.pkl', 'rb'))
y_mem = y_mem.T.flatten()
y_perc = y_perc.T.flatten()

av_mem, feature_names_mem = extract_basic(X_mem)
av_perc, feature_names_perc = extract_basic(X_perc)


def knn():
	knn = KNeighborsClassifier()
	print('mem:', cross_val_score(knn, av_mem, y_mem, cv=3))
	knn = KNeighborsClassifier()
	print('prec:', cross_val_score(knn, av_perc, y_perc, cv=3))


def random_forest():
	rf = RandomForestClassifier()
	print('mem:', cross_val_score(rf, av_mem, y_mem, cv=3))
	rf = RandomForestClassifier()
	print('prec:', cross_val_score(rf, av_perc, y_perc, cv=3))


def svm():
	svm = SVC()
	print('mem:', cross_val_score(svm, av_mem, y_mem, cv=3))
	svm = SVC()
	print('prec:', cross_val_score(svm, av_perc, y_perc, cv=3))


def mlp():
	mlp = MLPClassifier()
	print('mem:', cross_val_score(mlp, av_mem, y_mem, cv=3))
	mlp = MLPClassifier()
	print('prec:', cross_val_score(mlp, av_perc, y_perc, cv=3))


def main():
	knn()
	# random_forest()
	# svm()
	# mlp()

if __name__ == '__main__':
	main()