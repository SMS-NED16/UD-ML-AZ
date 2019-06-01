#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 09:57:00 2019

@author: saadmashkoor

ann_bank.py - Using Keras-based Aritifical Neural Network to predict churn rate
for bank customers.
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

# feature scaling is necessary for ANNs
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)