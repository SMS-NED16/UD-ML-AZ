#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:29:17 2019

@author: saadmashkoor
"""

"""simple_linear_regression.py
A Python program that implements univariate linear regression for salary data"""



"""DATA PREPROCESSING"""
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv('Salary_Data.csv')

# Examine the head of the dataset
# print(dataset.head())

# Extract features and target
X = dataset.iloc[:, :-1].values          # feature - YearsExperience, must be array, not vector!
y = dataset.iloc[:, 1].values           # target - Salary, can be a vecto

# Split dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# No need to standardise/normalize data because sklearn will do it for us in simple LR


"""TRAINING REGRESSION MODEL"""
# Import the LinearRegression estimator from the `linear_model` sublibrary in sklearn
from sklearn.linear_model import LinearRegression 

# Instantiate a LinearModel object
regressor = LinearRegression()

# Compute the coefficients for the linear model
regressor.fit(X_train, y_train)     # We have now created a model


"""TESTING REGRESSION MODEL BY PREDICTING TEST SET RESULTS"""
# Create a vector of salaries that the model will predict for the test data
y_pred = regressor.predict(X_test)


"""VISUALIZING THE TRAINING SET RESULTS"""
# Scatterplot
plt.scatter(x=X_train, y=y_train, c='red', label='Actual')

# Predictions made using the linear model 
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Predicted')

# Annotate plot
plt.title('Salary vs. Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)'); plt.grid(True); plt.legend()            
plt.show()                  # end of graph commands, render plot


"""VISUALIZE THE TEST SET RESULTS"""
# Create a new figure for the test set results
plt.figure()

# Scatterplot of test data
plt.scatter(X_test, y_test, c='red', label='Actual')

# Linear Model
# Regressor is already a unique model. Whether we plot it with the training 
# set or test makes no difference on the actual line because its coefficients
# have already been determined
plt.plot(X_test, y_pred, c='blue', label='Predicted')

# Annotate plot
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience'); plt.ylabel('Salary ($)')
plt.grid(True); plt.legend(); plt.show();


"""EXTRA MATERIAL - QUANTIZING ERROR"""
# Import methods to calculate mean absolute and squared errors b/w predictions and actual data
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Calculate errors
model_mae = mean_absolute_error(y_test, y_pred)
model_mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error- best choice b/c in same units as target variable
model_rmse = np.sqrt(mean_squared_error(y_test, y_pred))  

# Print results
print("Mean Absolute Error:\t" + str(model_mae) + "\nMean Squared Error:\t" + 
      str(model_mse) + "\nRoot of MS Error:\t" + str(model_rmse))
