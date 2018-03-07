# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE 

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

file = 'zimmermann.csv'

seeds = [100]
#seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#seeds = [(i+1)*100 for i in xrange(100)]

total_test = []
total_preds = []
d = {'0': [long(0), 0], '1': [0, 0]}
total_matrix = pd.DataFrame(data=d)

for seed in seeds:
	# Set random seed
	np.random.seed(seed)

	print "Seed " + str(seed)

	# Create a dataframe with the feature variables
	df = pd.read_csv(file)


	sm = SMOTE(random_state=seed)

	y = pd.factorize(df['Security'])[0]
	features = df.columns[1:-1]
	X_res, y_res = sm.fit_sample(df[features], y)

	X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.30, random_state=seed)

	# Show the number of observations for the test and training dataframes
	#print('Number of observations in the training data:', len(train))
	#print('Number of observations in the test data:',len(test))

	# View target
	#print y

	# Create a random forest Classifier. By convention, clf means 'Classifier'
	clf = RandomForestClassifier(n_jobs=2, random_state=seed)

	# Train the Classifier to take the training features and learn how they relate
	# to the training y (the species)
	clf.fit(X_train, y_train)

	preds = clf.predict(X_test)

	# View the PREDICTED species for the first five observations
	#print preds[0:5]

	total_test += y_test.tolist()
	total_preds += preds.tolist()

	#print clf.feature_importances_
	try:
		total_features
		j = 0
		for i in clf.feature_importances_:
			total_features[j] += i
			j += 1
	except NameError:
		total_features = np.copy(clf.feature_importances_)

		# Create confusion matrix
	matrix = pd.crosstab(y_test, preds, rownames=['Actual'], colnames=['Predicted'])
	total_matrix['0'][0] += matrix[0][0]
	total_matrix['0'][1] += matrix[0][1]
	try:
		total_matrix['1'][0] += matrix[1][0]
	except:
		print ''
	try:
		total_matrix['1'][1] += matrix[1][1]
	except:
		print ''

# View a list of the features and their importance scores
#print list(zip(train[features], clf.feature_importances_))
j = 0
for item in total_features:
	total_features[j] = total_features[j] / len(seeds)
	j += 1

total_matrix['0'][0] = total_matrix['0'][0] / len(seeds)
total_matrix['0'][1] = total_matrix['0'][1] / len(seeds)
total_matrix['1'][0] = total_matrix['1'][0] / len(seeds)
total_matrix['1'][1] = total_matrix['1'][1] / len(seeds)


print '\n\n\n\n'
print file

print ''
print 'Average Confusion Matrix:'

print total_matrix

print ''
print 'Feature Importance:'
print list(zip(X_train, total_features))

print ''
print 'Scores (Precision, Recall, FScore):'
print precision_recall_fscore_support(total_test, total_preds, average='binary')
print '\n\n'
