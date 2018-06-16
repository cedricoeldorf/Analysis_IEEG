from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.feature_selection import RFE
from collections import Counter
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


def make_feature_pickles(patient_name, patient_data, segment_patient_data, bin_size, with_overlap, overlap_step_size, use_multithreading_if_available, extract_frequency_data, frequency_band_mem, frequency_band_perc):
	if segment_patient_data:
		patient_data = segments_patient(patient_data, bin_size=bin_size, overlap=with_overlap, overlap_step=overlap_step_size, multithreaded=use_multithreading_if_available)
		print('done segmenting ')

	if extract_frequency_data:
		patient_data = extract_frequency(patient_data, frequency_band_mem, frequency_band_perc, multithreaded=use_multithreading_if_available)
		print('done extracting frequency bands')
	print(patient_data['eeg_m'].shape)
	if use_multithreading_if_available:
		features, feature_names = extract_multithreaded_basic(patient_data['eeg_m'])
	else:
		features, feature_names = extract_basic(patient_data['eeg_m'])
	pickle.dump((features, feature_names),
				open('preprocessed/pickle/features_mem_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(patient_name, segment_patient_data, bin_size, with_overlap, overlap_step_size, extract_frequency_data, frequency_band_mem, frequency_band_perc),
					 'wb'))

	if use_multithreading_if_available:
		features, feature_names = extract_multithreaded_basic(patient_data['eeg_p'])
	else:
		features, feature_names = extract_basic(patient_data['eeg_p'])
	pickle.dump((features, feature_names),
				open('preprocessed/pickle/features_perc_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(patient_name, segment_patient_data, bin_size, with_overlap, overlap_step_size, extract_frequency_data, frequency_band_mem, frequency_band_perc),
					 'wb'))


def compress(data, selectors):
	return [d for d, s in zip(data, selectors) if s]


def filter_features(features, y_mem, binned):
	y = y_mem.tolist()
	f = features.tolist()

	if binned:
		ind_to_remove = []
		features_to_keep = []
		n_leads = len(f)
		i = 0
		j = 0
		while len(features_to_keep) == 0:
			features_to_keep = [True for _ in range(len(f[i][j][0]))]
			if i < n_leads - 1:
				i += 1
			else:
				raise Exception('didnt find a lead with more than 0 features... RIP try different value for trial or bin')
		ymem_per_lead = [y.copy() for _ in range(n_leads)]
		for lead in range(n_leads):
			n_trials = len(f[lead])
			for trial in range(n_trials):
				n_bins = len(f[lead][trial])
				for bin in range(n_bins):
					feature_vector = f[lead][trial][bin]
					for i, feat in enumerate(feature_vector):
						if str(feat) == 'nan':
							features_to_keep[i] = False
					if len(feature_vector) == 0:
						ind_to_remove.append((lead, trial))
						break
		for i in sorted(ind_to_remove, reverse=True):
			del f[i[0]][i[1]]
			del ymem_per_lead[i[0]][i[1]]
		before = len(f[0][0][0])
		for lead in range(n_leads):
			n_trials = len(f[lead])
			for trial in range(n_trials):
				n_bins = len(f[lead][trial])
				for bin in range(n_bins):
					f[lead][trial][bin] = compress(f[lead][trial][bin], features_to_keep)
		print('Removed {} out of {} features'.format(before - len(f[0][0][0]), before))
		return np.array(f), np.array(ymem_per_lead)
	else:
		ind_to_remove = []
		features_to_keep = []
		n_leads = len(f)
		i = 0
		j = 0
		while len(features_to_keep) == 0:
			features_to_keep = [True for _ in range(len(f[i][j]))]
			if i < n_leads - 1:
				i += 1
			else:
				raise Exception('didnt find a lead with more than 0 features... RIP try different value for j')
		ymem_per_lead = [y.copy() for _ in range(n_leads)]
		for lead in range(n_leads):
			n_trials = len(f[lead])
			for trial in range(n_trials):
				feature_vector = f[lead][trial]
				for i, feat in enumerate(feature_vector):
					if str(feat) == 'nan':
						features_to_keep[i] = False
				if len(feature_vector) == 0:
					ind_to_remove.append((lead, trial))
		for i in sorted(ind_to_remove, reverse=True):
			del ymem_per_lead[i[0]][i[1]]
			del f[i[0]][i[1]]
		before = len(f[0][0])
		for lead in range(n_leads):
			n_trials = len(f[lead])
			for trial in range(n_trials):
				f[lead][trial] = compress(f[lead][trial], features_to_keep)
		print('Removed {} out of {} features'.format(before - len(f[0][0]), before))
		return np.array(f), np.array(ymem_per_lead)


def main():
	##############
	### params ###
	##############

	patient_name = 'raw_FAC002'

	segment_patient_data = True
	bin_size = 880
	with_overlap = False
	overlap_step_size = 220

	extract_frequency_data = False
	frequency_band_mem = 'theta'
	frequency_band_perc = 'alpha'

	use_multithreading_if_available = True

	run_prediction = True

	if not segment_patient_data:  # dont change this!!!
		bin_size = ''
		with_overlap = ''
		overlap_step_size = ''
	#####################
	### end of params ###
	#####################

	patient_data = load_raw(patient_name)
	if run_prediction:
		pickle_file = 'preprocessed/pickle/features_mem_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(patient_name, segment_patient_data, bin_size, with_overlap, overlap_step_size, extract_frequency_data, frequency_band_mem, frequency_band_perc)
		print('using file :', pickle_file)

		try:
			features, feature_names = pickle.load(open(pickle_file, 'rb'))
		except:
			print('pickles not found, making them now..')
			make_feature_pickles(patient_name, patient_data, segment_patient_data, bin_size, with_overlap, overlap_step_size, use_multithreading_if_available, extract_frequency_data, frequency_band_mem, frequency_band_perc)
			(features, feature_names) = pickle.load(open(pickle_file, 'rb'))

		features, patient_data['simVecM'] = filter_features(features, patient_data['simVecM'], segment_patient_data)
		print(features.shape, patient_data['simVecM'].shape)
		accuracies = [0 for _ in range(features.shape[0])]
		for lead in range(features.shape[0]):
			trials = np.array(features[lead])
			y_mem = np.array(patient_data['simVecM'][lead]).flatten()
			kfold = KFold(n_splits=3)
			model = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.01)
			results = cross_val_score(model, trials, y_mem, cv=kfold)
			accuracies[lead] = results.mean()
		print(accuracies)
		print(max(accuracies))
	else:
		make_feature_pickles(patient_name, patient_data, segment_patient_data, bin_size, with_overlap, overlap_step_size, use_multithreading_if_available, extract_frequency_data, frequency_band_mem, frequency_band_perc)

	'''
	10 - fold
	acc per lead: [0.6238095238095238, 0.6247619047619047, 0.565, 0.605, 0.5549999999999999, 0.5323809523809524, 0.5702380952380952, 0.5178571428571428, 0.41809523809523813, 0.545, 0.6209523809523809, 0.5402380952380952, 0.4819047619047619, 0.5607142857142857, 0.5638095238095239, 0.48071428571428576, 0.48666666666666664, 0.5507142857142857, 0.5347619047619048, 0.5354761904761904, 0.5016666666666667, 0.6133333333333333, 0.5742857142857142, 0.5995238095238096, 0.5742857142857143, 0.45690476190476187, 0.5011904761904763, 0.5354761904761905, 0.5900000000000001, 0.6064285714285715, 0.5597619047619048, 0.5609523809523809, 0.5321428571428571, 0.5311904761904762, 0.5357142857142857, 0.5345238095238095, 0.5404761904761906, 0.5604761904761906, 0.5595238095238095, 0.5707142857142857, 0.6835714285714285, 0.63, 0.6695238095238095, 0.5419047619047619, 0.5002380952380954, 0.49000000000000005, 0.47690476190476183, 0.4980952380952381, 0.4909523809523809, 0.5214285714285715, 0.5397619047619048, 0.6057142857142856, 0.5009523809523809, 0.5361904761904761, 0.5364285714285715, 0.48238095238095235, 0.5107142857142858, 0.5804761904761906, 0.5704761904761906, 0.5454761904761904, 0.5073809523809524, 0.49071428571428577, 0.5254761904761904, 0.4597619047619048, 0.5702380952380952, 0.5092857142857142, 0.545952380952381, 0.5502380952380952, 0.5611904761904762, 0.5902380952380952, 0.5595238095238095, 0.5897619047619047, 0.5164285714285713, 0.47095238095238096, 0.5302380952380952, 0.5464285714285714, 0.5064285714285715, 0.5502380952380952, 0.49071428571428566, 0.5752380952380952, 0.5357142857142857, 0.595, 0.6357142857142857, 0.536904761904762, 0.48047619047619056, 0.6402380952380953, 0.5995238095238096, 0.5552380952380952, 0.6009523809523809, 0.5611904761904761, 0.5902380952380952, 0.4969047619047619, 0.5804761904761906, 0.5421428571428573, 0.5407142857142857, 0.6297619047619047, 0.6211904761904762, 0.485952380952381, 0.4978571428571429, 0.536904761904762, 0.6549999999999999, 0.5116666666666666, 0.6047619047619047, 0.555952380952381, 0.5902380952380952, 0.5842857142857143, 0.5561904761904761, 0.5747619047619048, 0.5766666666666667, 0.6261904761904762, 0.6354761904761904, 0.549047619047619, 0.6788095238095238, 0.6121428571428571, 0.5561904761904762, 0.595, 0.5640476190476191, 0.5169047619047619, 0.5609523809523809, 0.5114285714285715, 0.5002380952380954, 0.5654761904761905, 0.555952380952381, 0.5407142857142857, 0.5597619047619048]
	best: 0.6835714285714285
	
	3 - fold
	[0.5027070529704419, 0.5123646473514779, 0.5447007387305894, 0.541849575651156, 0.48785484342990926, 0.5614574187884108, 0.4779045946736904, 0.48317237342698266, 0.49282996780801874, 0.49758560140474106, 0.47270997951419375, 0.47307579748317236, 0.4482733391864208, 0.49275680421422297, 0.566505706760316, 0.5567749487854843, 0.4680275095112672, 0.47797775826748606, 0.4877085162423178, 0.4877085162423178, 0.5323383084577115, 0.44344454199590283, 0.49758560140474106, 0.47827041264266895, 0.5565554580040971, 0.4926836406204273, 0.5420690664325432, 0.5221685689201053, 0.5271436932982148, 0.5369476148668423, 0.5469710272168569, 0.43832309043020184, 0.5663593795727245, 0.48266022827041266, 0.5026338893766461, 0.5224612232952882, 0.49290313140181446, 0.5174860989171788, 0.5073163593795726, 0.586113549897571, 0.5908691834942933, 0.4531021363769388, 0.5618232367573895, 0.5469710272168569, 0.5269973661106233, 0.5027802165642377, 0.5321919812701199, 0.46846649107404154, 0.5070968685981855, 0.6552531460345333, 0.5170471173544045, 0.5172666081357916, 0.5076821773485514, 0.586113549897571, 0.5271436932982148, 0.4730026338893767, 0.5567749487854844, 0.5417032484635645, 0.5025607257828505, 0.5122914837576822, 0.5027802165642377, 0.5074626865671642, 0.47278314310798947, 0.4725636523266023, 0.5172666081357916, 0.5321188176763243, 0.48266022827041266, 0.4580772607550483, 0.522241732513901, 0.48734269827333926, 0.5223148961076968, 0.5023412350014632, 0.4973661106233538, 0.48756218905472637, 0.5517266608135792, 0.5324114720515071, 0.5466052092478783, 0.44805384840503365, 0.47263681592039797, 0.5763827919227392, 0.531899326894937, 0.5197497361676466, 0.4678811823236757, 0.46312554872695344, 0.5126573017266608, 0.4679543459174715, 0.48287971905179977, 0.5128036289142522, 0.5117061750073163, 0.4776851038923032, 0.5318261633011413, 0.42376353526485216, 0.5276558384547849, 0.47307579748317236, 0.5172666081357916, 0.40393620134621017, 0.5272900204858062, 0.4781240854550775, 0.4732952882645596, 0.5119988293824992, 0.5465320456540824, 0.4531752999707346, 0.5516534972197834, 0.45815042434884407, 0.5272168568920105, 0.48763535264852215, 0.5073163593795726, 0.6008194322505122, 0.48763535264852215, 0.5023412350014632, 0.5070968685981855, 0.521729587357331, 0.4776851038923032, 0.5419959028387474, 0.571627158326017, 0.4924641498390401, 0.5269242025168276, 0.5075358501609598, 0.5122183201638864, 0.5369476148668423, 0.5173397717295873, 0.5320456540825286, 0.46341820310213633, 0.5025607257828505, 0.5074626865671642]
	0.6552531460345333
	
	3-fold
	'''


if __name__ == '__main__':
	main()

# EEG.shape (125, 203, 17, 879)
# features.shape (125, 203, 17)
