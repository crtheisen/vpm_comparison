from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn import linear_model
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

total_test = []
total_preds = []

precision_list = []
recall_list = []
fscore_list = []

avg_aoc = 0.0

d = {'0': [long(0), 0], '1': [0, 0]}
average_conf_matrix = pd.DataFrame(data=d)

for seed in seeds:
	print '1'
	# Set random seed and set classifier
	np.random.seed(seed)
	#print 'Seed Run: ' + str(seed)
	if args.classifier == 'rf':
		clf = RandomForestClassifier(n_estimators=50, max_features=.78,n_jobs=2, random_state=seed)
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

	y = train['Security']

	clf.fit(train[features], y)
	clf.predict(test[features])

	preds = clf.predict(test[features])

	#concatinate the runs together
	total_test += test['Security'].values.tolist()
	total_preds += preds.tolist()

	precision, recall, fscore, temp = precision_recall_fscore_support(test['Security'].values.tolist(), preds.tolist(), average='binary')
	precision_list.append(precision)
	recall_list.append(recall)
	fscore_list.append(fscore)

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

	avg_aoc = avg_aoc + roc_auc_score(test['Security'], preds)

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
print 'Avg AUC:'
print avg_aoc/int(args.seeds)



# with open('boxplot_data/precision/' + output_file + '_precision.csv', 'wb') as csv_file:
# 	writer = csv.writer(csv_file)
# 	for value in precision_list:
# 		writer.writerow([value])
#
# with open('boxplot_data/recall/' + output_file + '_recall.csv', 'wb') as csv_file:
# 	writer = csv.writer(csv_file)
# 	for value in recall_list:
# 		writer.writerow([value])
#
# with open('boxplot_data/fscore/' + output_file + '_fscore.csv', 'wb') as csv_file:
# 	writer = csv.writer(csv_file)
# 	for value in fscore_list:
# 		writer.writerow([value])
