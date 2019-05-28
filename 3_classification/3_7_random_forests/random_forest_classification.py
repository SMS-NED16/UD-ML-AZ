#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 06:31:40 2019

@author: saadmashkoor
"""

# import libraries
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# extract features and target
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# feature scaling - again not necessary but helps speed up visualization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# instantiate Random Forest Classifier 
from sklearn.ensemble import RandomForestClassifier
n_trees = 10
rf_classifier = RandomForestClassifier(n_estimators=n_trees, 
               criterion='entropy',random_state=0)
# High entropy - low homogeneity. We want to minimise entropy.
# Splitting stops when parent_entropy - child_entropy falls below threshold 
# Because this indicates homoegeneity.

# train the model
rf_classifier.fit(X_train, y_train)

# make predictions
y_pred = rf_classifier.predict(X_test)

# quantize error
from sklearn.metrics import confusion_matrix, classification_report
class_report = classification_report(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

# visualize training and test results
from visualize_classifier_results import visualize_results
train_title = "Random Forest Classification (" + str(n_trees) + " trees - Training)"
test_title = "Random Forest Classification (" + str(n_trees) + " trees - Test)"

plt.figure();
visualize_results(X_train, y_train, rf_classifier, train_title)

plt.figure();
visualize_results(X_test, y_test, rf_classifier, test_title) 


"""
Does not suffer from overfitting. There are fewer terminal leaves which has
improved classification performance because the results have been derived from
multiple trees. Instead of creating multiple small leaves for 0 class, the 
model has combined many such leaves into one node. 

However, this superleaf does not make sense in the context of test set. 
There are no 0 class data points in that region in the test set, and only 
multiple 1 class data points that are borderline misclassified. Overfitting
has been reduced but not substantially.

The conclusion? Gaussian Kernel SVM and Naive Bayes classifer are probably
the best classifiers for this data. They have created a smooth classification
boundary and don't tend to overfit the training data.

"""