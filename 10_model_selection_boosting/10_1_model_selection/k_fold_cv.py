#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 07:26:29 2019

@author: saadmashkoor

k_fold_cv.py - Implementing K-Fold Cross Validation for Social Network ad data
to improve accuracy and variance of Kernel SVM model
"""

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# extract features and labels
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# fit kernel SVM to the training set
from sklearn.svm import SVC
svc_classifier = SVC(kernel='rbf', random_state=0)
svc_classifier.fit(X_train, y_train)

# Predict using the kernel SVM classifier
y_pred = svc_classifier.predict(X_test)

# Evaluate performance using K-Fold Cross Validation

# Import the cross_val_Score function from `model_selection`
from sklearn.model_selection import cross_val_score

# store accuracies for each of 10 folds used in CV 
cv_accuracies = cross_val_score(svc_classifier, X=X_train, y=y_train, cv=10)

# find average accuracy and standard deviation of accuracies
acc_avg, acc_std = np.mean(cv_accuracies), np.std(cv_accuracies)

# low std, high accuracy -> low bias, low variance!