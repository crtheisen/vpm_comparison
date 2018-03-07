# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_recall_fscore_support

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

file = 'zimmermann_crashes-removed.csv'

#seeds = [100]
#seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
seeds = [(i+1)*100 for i in xrange(100)]

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

	# View the top 5 rows
	#print df.head()

	# Create a new column that for each row, generates a random number between 0 and 1, and
	# if that value is less than or equal to .70, then sets the value of that cell as True
	# and false otherwise. This is a quick and dirty way of randomly assigning some rows to
	# be used as the training data and some as the test data.
	df['is_train'] = np.random.uniform(0, 1, len(df)) <= .70

	# Create two new dataframes, one with the training rows, one with the test rows
	train, test = df[df['is_train']==True], df[df['is_train']==False]

	# Show the number of observations for the test and training dataframes
	#print('Number of observations in the training data:', len(train))
	#print('Number of observations in the test data:',len(test))

	# Create a list of the feature column's names
	features = df.columns[1:-2]

	# View features
	#print features

	y = pd.factorize(train['Security'])[0]

	# View target
	#print y

	# Create a random forest Classifier. By convention, clf means 'Classifier'
	clf = RandomForestClassifier(n_jobs=2, random_state=seed)

	# Train the Classifier to take the training features and learn how they relate
	# to the training y (the species)
	clf.fit(train[features], y)

	# Apply the Classifier we trained to the test data (which, remember, it has never seen before)
	clf.predict(test[features])

	# View the predicted probabilities of the first 10 observations
	#print clf.predict_proba(test[features])[0:10]

	 # Create actual english names for the plants for each predicted plant class
	preds = clf.predict(test[features])

	total_test += test['Security'].values.tolist()
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
	matrix = pd.crosstab(test['Security'], preds, rownames=['Actual'], colnames=['Predicted'])
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
print list(zip(train[features], total_features))

print ''
print 'Scores (Precision, Recall, FScore):'
print precision_recall_fscore_support(total_test, total_preds, average='binary')
print '\n\n'
