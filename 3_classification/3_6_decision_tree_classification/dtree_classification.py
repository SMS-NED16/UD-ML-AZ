#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 05:46:54 2019

@author: saadmashkoor
"""

"""dtree_classification.py - Decision Tree for classifying social media users into
potential buyers of an SUV."""

# import libraries
import pandas as pd

# import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# extract features and target
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values      # column, not array - otherwise index error in visualization

# train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

"""Since feature scaling is only required for euclidean distance, we don't really
need to do this. Furthermore, if we intend to plot actual decision trees, using
unscaled domain values will be useful for interpretability. However, our visualization
has a very high resolution, and the plotting procedure will be faster if we use scaled
features. This is why we're going to keep feature scaling, even though it is not
required for decision trees to model the data accurately."""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# instantiate decision tree classifier
from sklearn.tree import DecisionTreeClassifier

# Will use entropy as assessment criterion - measures quality of splits
# We want each node to be as homogeneous as possible, the more users in a node
# the lower the entropy. If entropy 0, then fully homogeneous node of users
# i.e. contains data points from only one class.
dtree_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)

# fit the model
dtree_classifier.fit(X_train, y_train)

# make predictions
y_pred = dtree_classifier.predict(X_test)

# quantize classification error
from sklearn.metrics import classification_report, confusion_matrix
class_report = classification_report(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

# visualize training/test set results
from visualize_classifier_results import visualize_results
train_title = "Decision Tree Classifier - Training"
test_title = "Decision Tree Classifier - Test"
visualize_results(X_train, y_train, dtree_classifier, train_title)
visualize_results(X_test, y_test, dtree_classifier, test_title)

"""
The visualization makes sense because the 2D domain has been split into
rectangular regions. The prediction boundary is composed of only horizontal and
vertical lines. It is making splits on the basis of independent variables. Each
rectangle represents a terminal leaf. 

There is evidence of overfitting: in the training set, the model has created
leaves for individual data points.

Peformance metrics show that the precision/f1 score/recall are around 90%, but
there is some overfitting. We could improve this with a random forest of
such decision trees.
"""