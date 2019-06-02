#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 08:29:35 2019

@author: saadmashkoor
"""

"""----------------------------DATA PREPROCESSING-----------------------------"""
# import libraries
import pandas as pd                         # will read CSV into a dataframe

# import dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# extracting features and labels 
X = dataset.iloc[:, 3:-1].values            # all except index, row num, id, name
y = dataset.iloc[:, -1].values              # did customer leave bank?

# encode categorical features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# create separate label encoder for each categorical variable
labelEncoder_country, labelEncoder_gender = LabelEncoder(), LabelEncoder()

# Fitting encoders
X[:, 1] = labelEncoder_country.fit_transform(X[:, 1])   # country
X[:, 2] = labelEncoder_gender.fit_transform(X[:, 2])    # gender

# One Hot encoding only for the country
one_hot_encoder = OneHotEncoder(categorical_features=[1])
X = one_hot_encoder.fit_transform(X).toarray() 

# Avoid dummy variable trap by dropping one dummy variable for country
X = X[:, 1:]

# splitting the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

"""---------------------------FITTING XG BOOST TO SET-----------------------------"""
# Classification problem so import a classifier
from xgboost import XGBClassifier

# Instantiate
xgb_classifier = XGBClassifier()

# Fit the classifier to the training set
xgb_classifier.fit(X_train, y_train)

# Predictions and evaluation
y_pred = xgb_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)

# K-Fold cross validation to improve prediction accuracy
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=xgb_classifier, X=X_train, y=y_train, cv=10)
acc_avg, acc_std = accuracies.mean(), accuracies.std()

"""Processed 10k samples in 3 seconds! 86.3% accuracy, 1.06% standard deviation"""