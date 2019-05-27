#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 06:39:45 2019

@author: saadmashkoor
"""

"""svm.py - Support Vector Machine classification"""

# import libraries
import pandas as pd

# import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# extract features and target
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# instantiate support vector machine classifier
from sklearn.svm import SVC
svm_classifier = SVC(kernel='linear', random_state=0) # linear decision boundary

# fit and train the classifier
svm_classifier.fit(X_train, y_train)

# make predictions using classifier
y_pred = svm_classifier.predict(X_test)

# visualize training set results
from visualize_classifier_results import visualize_results
visualize_results(X_train, y_train, svm_classifier, "SVM Classifier - Training Data");
visualize_results(X_test, y_test, svm_classifier, "SVM Classifier - Test Data")

# quantize classification performance
from sklearn.metrics import classification_report, confusion_matrix
class_report = classification_report(y_test, y_pred)
con_mat = confusion_matrix(y_test, y_pred)


"""The linear SVM classifier has an precision, recall and f1 score of 90%, 
while the KNN algo scored 93% for the same metrics. It is clearly better than
the logistic regression classifer, even though it also uses a linear decision
boundary.

The visualization of the training set shows there are quite a few 
data points that are misclassified. This could be a necessary cost to prevent
overfitting because the test data visualization shows fewer misclassified
examples, which means the decision boundary generalizes well to unseen data.
Even logistic regression had many misclassifications in the training
set, so this isn't entirely unexpected.

Gaussian kernel `rbf` creates a non-linear decision boundary which follows 
roughly the same path as the KNN boundary but is much smoother. It, too, has
93% classification precision and recall although its f1 score is lower."""