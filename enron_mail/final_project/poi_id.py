#!/usr/bin/python

import sys
reload(sys)
sys.setdefaultencoding('gbk')
import pickle
sys.path.append("../tools/")


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from plotScatter import plotScatter


def poi_email_ratio(from_poi_to_this_person, to_message):
    if from_poi_to_this_person or to_message == 'NaN':
        to_poi_ratio = 0
    else:
        to_poi_ratio = float(from_poi_to_this_person) / to_message
    return to_poi_ratio

def addValue(feature):
    if(features == 'NaN'):
        pass
        
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'to_poi_ratio', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'shared_receipt_with_poi', 'from_messages'] # You will need to use more features


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
del my_dataset['TOTAL']
for key in my_dataset:
    my_dataset[key]['to_poi_ratio'] = poi_email_ratio(my_dataset[key]['from_poi_to_this_person'], my_dataset[key]['to_messages'])


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# plotScatter(features, labels, 1, 0, 'shared_receipt_with_poi', 'Salary')
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# clf = GaussianNB()
# # clf = DecisionTreeClassifier()
# # clf = SVC()
# clf.fit(features, labels)

from sklearn import tree
from sklearn.model_selection import GridSearchCV
trees = tree.DecisionTreeClassifier()
parameters = {'min_samples_split' : range(5,80,5), 'splitter' : ('best', 'random')}

clf = GridSearchCV(trees, parameters)
clf.fit(features, labels)



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

print("the training is finished!!!")