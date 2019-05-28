#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 05:16:09 2019

@author: saadmashkoor
"""

"""naive_bayes.py - Naive Bayes classifier implementation in Python. Used to
classify social media users into potential buyers/non-buyers of an SUV."""

# import libraries
import pandas as pd

# import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# extract features and target
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# instantiate a naive bayes classifier
from sklearn.naive_bayes import GaussianNB
naive_bayes_classifier = GaussianNB() # No arguments, very simple

# train the model
naive_bayes_classifier.fit(X_train, y_train)

# make predictions
y_pred = naive_bayes_classifier.predict(X_test)

# visualize test set results
from visualize_classifier_results import visualize_results
train_title = "Naive Bayes Classification (Training)"
test_title = "Naive Bayes Classification (Test)"
visualize_results(X_train, y_train, naive_bayes_classifier, train_title)
visualize_results(X_test, y_test, naive_bayes_classifier, test_title)

# quantize performance
from sklearn.metrics import classification_report, confusion_matrix
class_report = classification_report(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)


"""
Naive Bayes created a non-linear classification boundary. The classification boundary 
looks very similar to that of the Gaussian kernel SVM. However, its classification 
metrics are poorer. Even though it has an overall precision and recall of 0.90, 
its recall for the positive class (1) is only 78%.

sklearn has automatically selected the optimal radius for our model. 
"""