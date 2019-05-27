#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:13:19 2019

@author: saadmashkoor
"""

"""kernel_svm.py - Python implementation of Kernel SVM algorithm with RBF kernel
to classify social media users into potential buyers/non-buyers of an SUV."""

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

# instantiate and train Kernel SVM classifier
from sklearn.svm import SVC
kernel_svc_classifier = SVC(kernel='rbf', random_state=0) # rbf, not linear kernel
kernel_svc_classifier.fit(X_train, y_train)

# make predictions with Kernel SVM classifier
y_pred = kernel_svc_classifier.predict(X_test)

# import visualization method and generate visualizations for train/test data
from visualize_classifier_results import visualize_results
visualize_results(X_train, y_train, kernel_svc_classifier, 'Kernel SVM (Default Penalty) - Train')
visualize_results(X_test, y_test, kernel_svc_classifier, 'Kernel SVM (Default Penalty) - Test')

# quantize classification results
from sklearn.metrics import classification_report, confusion_matrix
class_report = classification_report(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)