from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn import linear_model
from scipy.optimize import rosen, differential_evolution
from scipy.stats.stats import pearsonr
import pandas as pd
import numpy as np
import argparse
import csv

#Configuration
parser = argparse.ArgumentParser(
         description=(
             'Script for running comparisons of different VPMs.'
         )
      )
parser.add_argument(
        '-s', dest='seeds', default='1',
        choices=['1', '10', '100'],
        help='The number of crossfolds (or total seeds) to use for this run.'
    )
parser.add_argument(
        '-t', dest='train_size', default='.7',
        help='The size of the training set, expressed as a float between 0 and 1.'
    )
parser.add_argument(
        '-f', dest='file',
        required=True,
        help='The name of the .csv file with the target features and Security dependent variable.'
    )
parser.add_argument(
        '-c', dest='classifier', default='rf',
        choices=['rf', 'gnb', 'dtc', 'logr'],
        help='The type of classifier to be used for this run.'
    )
parser.add_argument(
        '-o', dest='output',
        required=True,
        help='The prefix of the output files for precision and recall values from the run for box plots.'
    )
args = parser.parse_args()

def tune_rf(params):

	n_estimators_ = int(params[0])
	max_features_ = params[1]
	max_leaf_nodes_ = max(int(params[2]), 2)
	min_samples_split_ = max(int(params[3]), 2)
	min_samples_leaf_ = max(int(params[4]), 1)

	file = args.file

	output_file = args.output

	if args.seeds == '10':
		seeds = [(i+1)*100 for i in xrange(10)]
	elif args.seeds == '100':
		seeds = [(i+1)*100 for i in xrange(100)]
	else:
		seeds = [100]

	train_size = float(args.train_size)
	if train_size > 1 or train_size < 0:
		print 'Train size invalid. Please enter a value between 0 and 1.'
		exit()

	avg_aoc = 0.0

	for seed in seeds:

		clf = RandomForestClassifier(max_leaf_nodes=max_leaf_nodes_,min_samples_split=min_samples_split_,
									 min_samples_leaf=min_samples_leaf_,
									 n_estimators=n_estimators_,
									 max_features=max_features_,
									 n_jobs=2)

		df = pd.read_csv(file)

		#create train/test
		df['is_train'] = np.random.uniform(0, 1, len(df)) <= train_size
		train, test = df[df['is_train']==True], df[df['is_train']==False]

		#set list of features
		features = df.columns[1:-2]

		#set dependent variable
		dep = 'Security'

		y = train['Security']

		clf.fit(train[features], y)
		clf.predict(test[features])

		preds = clf.predict(test[features])

		avg_aoc = avg_aoc + roc_auc_score(test['Security'], preds)

	#End for seed in seeds
	return (1-avg_aoc/int(args.seeds))

def tune_dtc(params):

	max_features_ = params[0]
	max_depth_ = max(int(params[1]), 2)
	min_samples_split_ = max(int(params[2]), 2)
	min_samples_leaf_ = max(int(params[3]), 1)

	file = args.file

	output_file = args.output

	if args.seeds == '10':
		seeds = [(i+1)*100 for i in xrange(10)]
	elif args.seeds == '100':
		seeds = [(i+1)*100 for i in xrange(100)]
	else:
		seeds = [100]

	train_size = float(args.train_size)
	if train_size > 1 or train_size < 0:
		print 'Train size invalid. Please enter a value between 0 and 1.'
		exit()

	avg_aoc = 0.0

	for seed in seeds:

		clf = DecisionTreeClassifier(min_samples_split=min_samples_split_,
									 min_samples_leaf=min_samples_leaf_,
									 max_features=max_features_,
									 max_depth=max_depth_)

		df = pd.read_csv(file)

		#create train/test
		df['is_train'] = np.random.uniform(0, 1, len(df)) <= train_size
		train, test = df[df['is_train']==True], df[df['is_train']==False]

		#set list of features
		features = df.columns[1:-2]

		#set dependent variable
		dep = 'Security'

		y = train['Security']

		clf.fit(train[features], y)
		clf.predict(test[features])

		preds = clf.predict(test[features])

		avg_aoc = avg_aoc + roc_auc_score(test['Security'], preds)

	#End for Seed in Seeds
	return (1-avg_aoc/int(args.seeds)) #scipy DE minimizes functions; need to take inverse

def tune_logr(params):

	C_ = params[0]

	file = args.file

	output_file = args.output

	if args.seeds == '10':
		seeds = [(i+1)*100 for i in xrange(10)]
	elif args.seeds == '100':
		seeds = [(i+1)*100 for i in xrange(100)]
	else:
		seeds = [100]

	train_size = float(args.train_size)
	if train_size > 1 or train_size < 0:
		print 'Train size invalid. Please enter a value between 0 and 1.'
		exit()

	avg_aoc = 0.0

	for seed in seeds:

		clf = linear_model.LogisticRegression(C=C_, solver='saga')

		df = pd.read_csv(file)

		#create train/test
		df['is_train'] = np.random.uniform(0, 1, len(df)) <= train_size
		train, test = df[df['is_train']==True], df[df['is_train']==False]

		#set list of features
		features = df.columns[1:-2]

		#set dependent variable
		dep = 'Security'

		y = train['Security']

		clf.fit(train[features], y)
		clf.predict(test[features])

		preds = clf.predict(test[features])

		avg_aoc = avg_aoc + roc_auc_score(test['Security'], preds)

	#End for Seed in Seeds
	return (1-avg_aoc/int(args.seeds)) #scipy DE minimizes functions; need to take inverse

def tune_gnb():

	file = args.file

	output_file = args.output

	if args.seeds == '10':
		seeds = [(i+1)*100 for i in xrange(10)]
	elif args.seeds == '100':
		seeds = [(i+1)*100 for i in xrange(100)]
	else:
		seeds = [100]

	train_size = float(args.train_size)
	if train_size > 1 or train_size < 0:
		print 'Train size invalid. Please enter a value between 0 and 1.'
		exit()

	avg_aoc = 0.0

	for seed in seeds:

		clf = GaussianNB()

		df = pd.read_csv(file)

		#create train/test
		df['is_train'] = np.random.uniform(0, 1, len(df)) <= train_size
		train, test = df[df['is_train']==True], df[df['is_train']==False]

		#set list of features
		features = df.columns[1:-2]

		#set dependent variable
		dep = 'Security'

		y = train['Security']

		clf.fit(train[features], y)
		clf.predict(test[features])

		preds = clf.predict(test[features])

		avg_aoc = avg_aoc + roc_auc_score(test['Security'], preds)

	#End for seed in seeds
	return avg_aoc/int(args.seeds)

def tune_cnb(params):

	alpha_ = params[0]

	file = args.file

	output_file = args.output

	if args.seeds == '10':
		seeds = [(i+1)*100 for i in xrange(10)]
	elif args.seeds == '100':
		seeds = [(i+1)*100 for i in xrange(100)]
	else:
		seeds = [100]

	train_size = float(args.train_size)
	if train_size > 1 or train_size < 0:
		print 'Train size invalid. Please enter a value between 0 and 1.'
		exit()

	avg_aoc = 0.0

	for seed in seeds:

		clf = ComplementNB(alpha=alpha_)

		df = pd.read_csv(file)

		#create train/test
		df['is_train'] = np.random.uniform(0, 1, len(df)) <= train_size
		train, test = df[df['is_train']==True], df[df['is_train']==False]

		#set list of features
		features = df.columns[1:-2]

		#set dependent variable
		dep = 'Security'

		y = train['Security']

		clf.fit(train[features], y)
		clf.predict(test[features])

		preds = clf.predict(test[features])

		avg_aoc = avg_aoc + roc_auc_score(test['Security'], preds)

	#End for Seed in Seeds
	return (1-avg_aoc/int(args.seeds)) #scipy DE minimizes functions; need to take inverse



#bounds = (50, .78, 50, 2, 1)
#print 'RF AOC: ' + tune_rf(bounds) #Defaults

####

#bounds = [(50, 150), (0.01, 1), (2, 50), (2, 20), (1, 20)]
#result = differential_evolution(tune_rf, bounds, maxiter=20)

#bounds = [ (0.01, 1), (2, 50), (2, 20), (1, 20)]
#result = differential_evolution(tune_dtc, bounds, maxiter=50)

# bounds = [(0.001, 1000)]
# result = differential_evolution(tune_logr, bounds, maxiter=50)

bounds = [(0.0, 5.0)]
result = differential_evolution(tune_cnb, bounds, maxiter=50)

print result.x, result.fun, (1-result.fun) #inverse back for actual AUC value

####

#result = tune_gnb()
#print result

#rforest: [89.6417629   0.71362012 38.49527865  5.73602815  2.24500321] 0.4494534919450419 0.5505465080549581
#rforest: [76.44702851  0.79024905 35.19906614 17.67531563  2.24788728] 0.4278069629732033 0.5721930370267967

#rforest x20: [71.61479235  0.87743191 49.87772713  3.76237584  1.91823076] 0.40888731730315886 0.5911126826968411

#dtc [ 0.55964945 45.77263984  6.45227521 13.95019508] 0.3758901037286234 0.6241098962713766
#dtc [ 0.85170846 33.12007924 13.86882716  2.62479165] 0.375970744680851 0.624029255319149

#dtc x10 [ 0.79678206 35.09696238  3.04028864  1.36665915] 0.3561333333333333 0.6438666666666667
#dtc x10 [ 0.58252669 18.54203745 12.59643033  3.48078034] 0.34768434853477914 0.6523156514652209

#dtc x50 [ 0.87991989 34.57507064  6.77095133  3.66539779] 0.3083292808126654 0.6916707191873346
#dtc x50 [ 0.29399273 29.54318172 13.7232189   1.87553234] 0.3075080828643276 0.6924919171356724

#logr: [978.32206384] 0.5 0.5

#logr x50 [736.82818869] 0.5 0.5

#gnb 0.6088320864505403

#cnb [3.42525775] 0.33964125857164906 0.6603587414283509
#cnb [4.18890433] 0.2931913685406019 0.7068086314593981

#cnb x50 [0.6668177] 0.25116603134752924 0.7488339686524708
