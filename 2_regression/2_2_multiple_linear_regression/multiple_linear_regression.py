#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:17:30 2019

@author: saadmashkoor
"""

"""PREPROCESSING"""
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv('50_Startups.csv')

"""The dataset consists of marketing, r&d, and admin expenditure as well as the
state of 50 startups. Our target variable is the startup's profit. Our features
are the three types of expenditure (numerical) and the state (categorical)."""

# Extracting features and target data
X = dataset.iloc[:, :-1].values  # all rows for all columns except last one (target)
y = dataset.iloc[:, 4].values   # all rows for the last column

# Encoding categorical data 
# First convert State column from categorical to numerical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X = LabelEncoder()
X[:, 3] = label_encoder_X.fit_transform(X[:, 3])

# Use one hot encoding to convert to non relational flags
one_hot_encoder = OneHotEncoder(categorical_features=[3])
X =  one_hot_encoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap - sklearn would do this automatically but good for concepts
X = X[:, 1:] # Removed the first column from X. This removes an extra dummy variable.

# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


"""TRAINING - MULTIPLE LINEAR REGRESSION MODEL"""
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()          # insantiate estimator
regressor.fit(X_train, y_train)         # compute parameters

"""TESTING - MULTIPLE LINEAR REGRESSION MODEL"""
y_pred = regressor.predict(X_test)


"""QUANTIZING ERROR"""
from sklearn.metrics import mean_absolute_error, mean_squared_error
print("Mean Squared Error =\t" + str(mean_squared_error(y_test, y_pred)))
print("Mean Absolute Error =\t" + str(mean_absolute_error(y_test, y_pred)))
print("RMS Error =\t\t" + str(np.sqrt(mean_squared_error(y_test, y_pred))))


"""BACKWARD ELIMINATION TO DETERMINE OPTIMAL MODEL
Will contain only statistically significant variables in the model
"""
# Building optimal model using backward elimination
import statsmodels.formula.api as sm

# Associate a placeholder variable with b0 (x0). This is 1 for all examples.
# Needs to be added explicitly for statsmodels
# Add a column of ones to the beginning of the features matrix
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

# Optimal feature subset - initially has all the possible predictors
X_opt = X[:, [0, 1, 2, 3, 4, 5]]

# Fit a multiple linear regression model to the new feature matrix
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() # OLS = Ordinary Least Squares

# Find and eliminate predictor with the highest p value
print(regressor_OLS.summary())

# We can see that x2 variable has p value of 99% - remove it from features
X_opt = X[:, [0, 1, 3, 4, 5]]

# Fit the model without this variable
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() 
print(regressor_OLS.summary())

# Iterating - eliminatign X[1]
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() 
print(regressor_OLS.summary())

# Iterating - eliminating X[4]
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() 
print(regressor_OLS.summary())

# X[5] (marketing expenditure ) has p value slightly above 5%
# Iterating - 
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit() 
print(regressor_OLS.summary())

# Final model consists of a constant term and R&D expenditure
# The optimal subset of variables contains only ONE independent variable
# This means we did not necessarily need to make a multivariate model in the first place
