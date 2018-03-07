from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn import linear_model
from scipy.stats.stats import pearsonr
import pandas as pd
import numpy as np
import argparse

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
args = parser.parse_args()

#file = 'shin.csv'
file = args.file

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

total_test = []
total_preds = []
d = {'0': [long(0), 0], '1': [0, 0]}
average_conf_matrix = pd.DataFrame(data=d)

#print 'Running ' + file + ' at ' + str(len(seeds)) + ' cross fold validation.'

for seed in seeds:
	# Set random seed and set classifier
	np.random.seed(seed)
	#print 'Seed Run: ' + str(seed)
	if args.classifier == 'rf':
		clf = RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=seed)
	elif args.classifier == 'gnb':
		clf = GaussianNB()
	elif args.classifier == 'dtc':
		clf = DecisionTreeClassifier()
	elif args.classifier == 'logr':
		clf = linear_model.LogisticRegression()
	else:
		print 'Classifier invalid. Please enter a valid classifier.'
		exit()

	df = pd.read_csv(file)

	#create train/test
	df['is_train'] = np.random.uniform(0, 1, len(df)) <= train_size
	train, test = df[df['is_train']==True], df[df['is_train']==False]

	#set list of features
	features = df.columns[1:-2]

	#set dependent variable
	dep = 'Security'

	#get pearson for all features, list top 20
	ret = []
	for column in features:
		ret.append([column, pearsonr(df[column].values.tolist(), df[dep].values.tolist())[0]])

	#print sorted(ret, key=lambda x:x[1], reverse=True)[:20]

	y = train['Security']
	#y = pd.factorize(train['Security'])[0]

	#clf = RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=seed)
	#clf = GaussianNB()
	#clf = DecisionTreeClassifier()
	#clf = linear_model.LogisticRegression()

	clf.fit(train[features], y)
	clf.predict(test[features])

	preds = clf.predict(test[features])

	#concatinate the runs together
	total_test += test['Security'].values.tolist()
	total_preds += preds.tolist()

	#feature importance (1 per run)
	# print sorted(clf.feature_importances_, reverse=True)[:20]
	# print sum(clf.feature_importances_)
	# try:
	# 	total_features
	# 	j = 0
	# 	for i in clf.feature_importances_:
	# 		total_features[j] += i
	# 		j += 1
	# except NameError:
	# 	total_features = np.copy(clf.feature_importances_)



	#Add data to total average confusion matrix
	matrix = pd.crosstab(test['Security'], preds, rownames=['Actual'], colnames=['Predicted'])
	average_conf_matrix['0'][0] += matrix[0][0]
	average_conf_matrix['0'][1] += matrix[0][1]
	try:
		average_conf_matrix['1'][0] += matrix[1][0]
	except:
		print ''
	try:
		average_conf_matrix['1'][1] += matrix[1][1]
	except:
		print ''

#End for seed in seeds

average_conf_matrix['0'][0] = average_conf_matrix['0'][0] / len(seeds)
average_conf_matrix['0'][1] = average_conf_matrix['0'][1] / len(seeds)
average_conf_matrix['1'][0] = average_conf_matrix['1'][0] / len(seeds)
average_conf_matrix['1'][1] = average_conf_matrix['1'][1] / len(seeds)

#Print Results

print ''
print '***Settings***'
print 'Target: ' + file
print 'Crossfolds: ' + str(args.seeds)
print 'Training Set Size: ' + str(args.train_size)
print 'Classifier: ' + str(args.classifier)

print ''
print 'Average Confusion Matrix:'
print average_conf_matrix

print ''
print 'Scores (Precision, Recall, FScore):'
print precision_recall_fscore_support(total_test, total_preds, average='binary')
print ''