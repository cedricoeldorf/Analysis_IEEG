# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# import xgboost as xgb
# from sklearn.feature_selection import RFE
# from collections import Counter
from load_raw import *
from separate_leads import *

np.random.seed(0)


def knn():
	knn = KNeighborsClassifier(n_jobs=-1)
	parameters = {'leaf_size': np.arange(15, 45, 5), 'n_neighbors': np.arange(2, 25, 1)}
	clf = GridSearchCV(knn, parameters)


# clf.fit(av_mem, y_mem)
# print(clf.cv_results_)


# print('mem:', cross_val_score(knn, av_mem, y_mem, cv=3))
# knn = KNeighborsClassifier()
# print('perc:', cross_val_score(knn, av_perc, y_perc, cv=3))


def random_forest():
	rf = RandomForestClassifier()


# print('mem:', cross_val_score(rf, av_mem, y_mem, cv=3))
# rf = RandomForestClassifier()
# print('perc:', cross_val_score(rf, av_perc, y_perc, cv=3))


def svm():
	svm = SVC()


# print('mem:', cross_val_score(svm, av_mem, y_mem, cv=3))
# svm = SVC()
# print('prec:', cross_val_score(svm, av_perc, y_perc, cv=3))


def mlp():
	mlp = MLPClassifier()


# print('mem:', cross_val_score(mlp, av_mem, y_mem, cv=3))
# mlp = MLPClassifier()
# print('perc:', cross_val_score(mlp, av_perc, y_perc, cv=3))


def xgboost():
	print('TODO: FIX THIS CODE -> old code... doesnt work anymore!!!')
	return


################
## W-I-P Cedric
# feat_info = input("Would you like to get feature info? (y/n) ")
#
# if feat_info == 'y':
# 	gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.01)
#
# 	print("Perception...")
# 	selector = RFE(gbm, step=100, verbose=2)
# 	# selector.fit(av_perc, y_perc)
# 	feats = selector.support_
# 	feats_ind = [i for i, x in enumerate(feats) if x]
# 	selected_features = [feature_names_perc[i] for i in (feats_ind)]
# 	lead_freq = [s[-3:] for s in selected_features]
# 	lead_freq = [s.replace('_', '') for s in lead_freq]
# 	lead_freq = [int(s.replace('d', '')) for s in lead_freq]
# 	counts = Counter(np.sort(lead_freq))
# 	leads = set(lead_freq)
# 	# freq = [len(lead_freq) for key, group in groupby(lead_freq)]
# 	plt.bar(counts.keys(), counts.values())
# 	plt.ion()
# 	plt.title("Perception")
# 	plt.show()
# 	av_perc = selector.transform(av_perc)
#
# 	print("Memory...")
# 	gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.01)
# 	selector = RFE(gbm, step=100, verbose=2)
# 	selector.fit(av_mem, y_mem)
# 	feats = selector.support_
# 	feats_ind = [i for i, x in enumerate(feats) if x]
# 	selected_features = [feature_names_perc[i] for i in (feats_ind)]
# 	lead_freq = [s[-3:] for s in selected_features]
# 	lead_freq = [s.replace('_', '') for s in lead_freq]
# 	lead_freq = [int(s.replace('d', '')) for s in lead_freq]
# 	counts = Counter(np.sort(lead_freq))
# 	leads = set(lead_freq)
# 	# freq = [len(lead_freq) for key, group in groupby(lead_freq)]
# 	plt.bar(counts.keys(), counts.values())
# 	plt.ion()
# 	plt.title("Memory")
# 	plt.show()
# 	av_mem = selector.transform(av_mem)
#
# 	gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.01)
# 	print('perc:', cross_val_score(gbm, av_perc, y_perc, cv=3))
# 	gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.01)
# 	print('mem:', cross_val_score(gbm, av_mem, y_mem, cv=3))
#
# else:
# 	gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.01)
# 	print('mem:', cross_val_score(gbm, av_mem, y_mem, cv=3))
#
# 	gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.01)
# 	print('perc:', cross_val_score(gbm, av_perc, y_perc, cv=3))


'''
		patient_data = contains all the data for patient X

		patient_data['eeg_m'] = all the memory eeg leads shape = L leads by T trials by D data points
		patient_data['eeg_p'] = all the perc eeg leads shape = L leads by T trials by D data points

		patient_data['simVecM'] # all the memory y values shape = T trials
		patient_data['simVecP'] # all the perception y values shape = T trials
'''


def main():
	##############
	### params ###
	##############

	patient = 'raw_FAC002'

	segment_patient_data = False
	bin_size = 880
	with_overlap = False
	overlap_step_size = 220

	extract_frequency_data = True
	frequency_band_mem = 'theta'
	frequency_band_perc = 'alpha'

	use_multithreading_if_available = True

	#####################
	### end of params ###
	#####################

	if not segment_patient_data:  # dont change this!!!
		bin_size = ''
		with_overlap = ''
		overlap_step_size = ''

	patient_data = load_raw(patient)

	if segment_patient_data:
		patient_data = segments_patient(patient_data, bin_size=bin_size, overlap=with_overlap, overlap_step=overlap_step_size, multithreaded=use_multithreading_if_available)
		print('done segmenting ')

	if extract_frequency_data:
		patient_data = extract_frequency(patient_data, frequency_band_mem, frequency_band_perc, multithreaded=use_multithreading_if_available)
		print('done extracting frequency bands')

	if use_multithreading_if_available:
		features, feature_names = extract_multithreaded_basic(patient_data['eeg_m'])
	else:
		features, feature_names = extract_basic(patient_data['eeg_m'])
	pickle.dump((features, feature_names), open('preprocessed/pickle/features_mem_{}_{}_{}_{}_{}_{}_{}.pkl'.format(patient, segment_patient_data, bin_size, with_overlap, overlap_step_size, frequency_band_mem, frequency_band_perc), 'wb'))

	if use_multithreading_if_available:
		features, feature_names = extract_multithreaded_basic(patient_data['eeg_p'])
	else:
		features, feature_names = extract_basic(patient_data['eeg_p'])
	pickle.dump((features, feature_names), open('preprocessed/pickle/features_perc_{}_{}_{}_{}_{}_{}_{}.pkl'.format(patient, segment_patient_data, bin_size, with_overlap, overlap_step_size, frequency_band_mem, frequency_band_perc), 'wb'))

	# accuracies = [0 for _ in range(features.shape[0])]
	# for lead in range(features.shape[0]):
	# 	trials = features[lead]
	# 	y_mem = patient_data['simVecM'].flatten()
	# 	kfold = KFold(n_splits=10)
	# 	model = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.01)
	#
	# 	results = cross_val_score(model, trials, y_mem, cv=kfold)
	#
	# 	accuracies[lead] = results.mean()
	# 	print("Accuracy lead {}: %.2f%%".format(lead) % (accuracy * 100.0))
	# print(accuracies)
	# print(max(accuracies))

	'''
	acc per lead: [0.6238095238095238, 0.6247619047619047, 0.565, 0.605, 0.5549999999999999, 0.5323809523809524, 0.5702380952380952, 0.5178571428571428, 0.41809523809523813, 0.545, 0.6209523809523809, 0.5402380952380952, 0.4819047619047619, 0.5607142857142857, 0.5638095238095239, 0.48071428571428576, 0.48666666666666664, 0.5507142857142857, 0.5347619047619048, 0.5354761904761904, 0.5016666666666667, 0.6133333333333333, 0.5742857142857142, 0.5995238095238096, 0.5742857142857143, 0.45690476190476187, 0.5011904761904763, 0.5354761904761905, 0.5900000000000001, 0.6064285714285715, 0.5597619047619048, 0.5609523809523809, 0.5321428571428571, 0.5311904761904762, 0.5357142857142857, 0.5345238095238095, 0.5404761904761906, 0.5604761904761906, 0.5595238095238095, 0.5707142857142857, 0.6835714285714285, 0.63, 0.6695238095238095, 0.5419047619047619, 0.5002380952380954, 0.49000000000000005, 0.47690476190476183, 0.4980952380952381, 0.4909523809523809, 0.5214285714285715, 0.5397619047619048, 0.6057142857142856, 0.5009523809523809, 0.5361904761904761, 0.5364285714285715, 0.48238095238095235, 0.5107142857142858, 0.5804761904761906, 0.5704761904761906, 0.5454761904761904, 0.5073809523809524, 0.49071428571428577, 0.5254761904761904, 0.4597619047619048, 0.5702380952380952, 0.5092857142857142, 0.545952380952381, 0.5502380952380952, 0.5611904761904762, 0.5902380952380952, 0.5595238095238095, 0.5897619047619047, 0.5164285714285713, 0.47095238095238096, 0.5302380952380952, 0.5464285714285714, 0.5064285714285715, 0.5502380952380952, 0.49071428571428566, 0.5752380952380952, 0.5357142857142857, 0.595, 0.6357142857142857, 0.536904761904762, 0.48047619047619056, 0.6402380952380953, 0.5995238095238096, 0.5552380952380952, 0.6009523809523809, 0.5611904761904761, 0.5902380952380952, 0.4969047619047619, 0.5804761904761906, 0.5421428571428573, 0.5407142857142857, 0.6297619047619047, 0.6211904761904762, 0.485952380952381, 0.4978571428571429, 0.536904761904762, 0.6549999999999999, 0.5116666666666666, 0.6047619047619047, 0.555952380952381, 0.5902380952380952, 0.5842857142857143, 0.5561904761904761, 0.5747619047619048, 0.5766666666666667, 0.6261904761904762, 0.6354761904761904, 0.549047619047619, 0.6788095238095238, 0.6121428571428571, 0.5561904761904762, 0.595, 0.5640476190476191, 0.5169047619047619, 0.5609523809523809, 0.5114285714285715, 0.5002380952380954, 0.5654761904761905, 0.555952380952381, 0.5407142857142857, 0.5597619047619048]
	best: 0.6835714285714285

	'''


if __name__ == '__main__':
	main()
