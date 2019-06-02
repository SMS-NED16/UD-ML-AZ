#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 07:53:22 2019

@author: saadmashkoor

grid_search.py - Using Grid Search to identify optimal hyperparameters for a 
kernel SVM classification model in conjunction with a 
"""

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

# Applying Grid Search to find the best model and the best hyperparameters
from sklearn.model_selection import GridSearchCV

# Create a dictionary where keys are params to optimise, vals are lists of 
# values that grid search will use to find optimal value
parameters = [
            { 'C': [1, 10, 100, 1000], 'kernel': ['linear']},   # linear model params
            { 'C': [1, 10, 100, 1000], 'kernel': ['rbf'],       # non-linear model params 
             'gamma': [0.5, 0.1, 0.01, 0.001]}]
 
# Perform Grid search for the specified options to identify optimal values for hyperparams
grid_search = GridSearchCV(estimator=svc_classifier, param_grid=parameters, 
                           scoring='accuracy', cv=10, n_jobs=-1)

# Fit the model to the training set
grid_search = grid_search.fit(X_train, y_train)

# Get best model from the grid_search object
best_model = grid_search.best_estimator_

# Use this model to make predictions
y_pred_best = best_model.predict(X_test)

# confusion matrix for best model
from sklearn.metrics import confusion_matrix
conf_mat_best = confusion_matrix(y_test, y_pred_best) 
best_accuracy = grid_search.best_score_

