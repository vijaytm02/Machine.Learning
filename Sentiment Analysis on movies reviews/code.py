from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
import numpy as np
import csv
import sys


def print_GridCV_scores(gs_clf, export_file):
	with open(export_file, 'w') as outfile:
		csvwriter = csv.writer(outfile, delimiter=',')

		# Create the header using the parameter names 
		header = ["mean","std"]
		param_names = [param for param in gs_clf.param_grid]
		header.extend(param_names)

		csvwriter.writerow(header)

		for config in gs_clf.grid_scores_:
			# Get mean and standard deviation
			mean = config[1]
			std = np.std(config[2])
			row = [mean,std]

			# Get the list of parameter settings and add to row
			params = [str(p) for p in config[0].values()]
			row.extend(params)

			csvwriter.writerow(row)

X_train = []
y_train = []

ID_test = []
X_test = []

# Get training data
with open('train.tsv', 'r') as f:
	f.readline()
	csvreader = csv.reader(f, delimiter='\t')
	for row in csvreader:
		X_train.append(row[2])
		y_train.append(row[3])

# Get test data:
with open('test.tsv', 'r') as f:
	f.readline()
	csvreader = csv.reader(f, delimiter='\t')
	for row in csvreader:
		ID_test.append(row[0])
		X_test.append(row[2])

# Use word and character features
words = TfidfVectorizer(analyzer="word", binary=False, use_idf=True, stop_words="english", min_df=3)
char = TfidfVectorizer(analyzer="char", binary=False, use_idf=True)

# Use percentile-based feature selection
select = SelectPercentile(score_func=chi2)

# Stack the features together
feat = FeatureUnion([('words', words),
	                 ('char', char)
])

# Construct transformation pipeline
text_clf = Pipeline([('feat', feat),
	                 # ('select', select),
                     # ('clf', MultinomialNB()),
                     ('clf', SGDClassifier(penalty='l2'))
])

# Set the parameters to be optimized in the Grid Search
parameters = {'feat__words__ngram_range': [(1,5), (1,6)],
			  # 'feat__words__stop_words': ("english", None),
              'feat__words__min_df': (2,3),
              'feat__words__use_idf': (True, False),
              'feat__char__use_idf': (True, False),
              # 'select__percentile': (20, 30),
              # For SGD
              'clf__alpha': (.00001, .000001),
              # 'clf__penalty': ('l2', 'l1', 'elasticnet'),
              'clf__loss': ("hinge", "log", "modified_huber")
}

# Fit the grid search using max CPU
gs_clf = GridSearchCV(text_clf, parameters, cv=2, verbose=True, n_jobs=-2)

# Transform and fit the model
gs_clf.fit(X_train, y_train)

# Transform the test data and get predictions
predictions = gs_clf.predict(X_test)

# Print GridCV scores to file
print_GridCV_scores(gs_clf, 's9.scores.csv')

# Get the best parameters and print them so we know
best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

# Sanity check: make sure that the predictions array is the correct length
if len(ID_test) != len(predictions):
	raise StandardError("Test data error")
	sys.exit()

# Create the submission file
with open('s9.csv', 'w') as outfile:
	outfile.write("PhraseId,Sentiment\n")
	for phrase_id,pred in zip(ID_test,predictions):
		outfile.write('{},{}\n'.format(phrase_id,pred))
