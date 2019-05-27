#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 05:19:05 2019

@author: saadmashkoor
"""

"""knn.py - K-Nearest Neighbour for classification of new users into
potential buyers/non-buyers based on Social Media Ad data"""

# import libraries
import numpy as np
import pandas as pd

# import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# extract features and target variable
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# instantiate and train a KNN classification model
from sklearn.neighbors import KNeighborsClassifier

# default number of neighbours with default distance computation
knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)    

# fit and train the KNN model
knn_classifier.fit(X_train, y_train)

# make predictions for the test set
y_pred = knn_classifier.predict(X_test)

# import and call function to visualize classification results
from visualize_classifier_results import visualize_results
visualize_results(X_test, y_test, knn_classifier, 'KNN Classification (5 Neighbours)')

# quantize test set performance using classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
con_mat = confusion_matrix(y_test, y_pred)
class_rep = classification_report(y_test, y_pred)

"""Classification report and confusion matrix both show that the KNN model
has performed very well. It only misclassified 7% of TNs as positives, and 
detected 93% of all the TPs in the dataset. It's weighted F1 score is also 93%.

The prediction boundary/decision boundary is very non-linear. Logistic regression
predicted a linear boundary and thus had higher misclassification rate. The non-linearity
allows the KNN algo to better classify the data.

Default parameters are chosen to prevent overfitting. 
"""