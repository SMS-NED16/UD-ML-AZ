#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:26:22 2019

@author: saadmashkoor
"""

"""simple_linear_regression_rpt.py
Reconstructing the simple linear regression model for practice"""

"""PREPROCESSING"""
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv('Salary_Data.csv')

# Extract features and target variable
X = dataset['YearsExperience'].values.reshape(-1, 1)    # must be array
y = dataset['Salary'].values                            # can be a vector

# Split the dataset into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)


"""CREATING AND TRAINING A LINEAR MODEL"""
from sklearn.linear_model import LinearRegression

# Instantiate model
regressor = LinearRegression()

# Fit the object to the given dataset to create new model with required params
regressor.fit(X_train, y_train)


"""TESTING LINEAR MODEL"""
y_pred = regressor.predict(X_test)


"""VISUALIZING RESULTS - TRAINING SET"""
plt.scatter(X_train, y_train, color='red', label='Actual')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Predicted')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Experience (years)'); plt.ylabel('Salary ($)')
plt.grid(True); plt.legend(), plt.show()

"""VISUALIZING RESULTS - TEST SET"""
plt.figure()
plt.scatter(X_test, y_test, color='red', label='Actual')
plt.plot(X_test, y_pred, color='blue', label='Predicted')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Experience (years)'); plt.ylabel('Salary ($)')
plt.grid(True); plt.legend(), plt.show()


"""QUANTIZING ERROR"""
from sklearn.metrics import mean_absolute_error, mean_squared_error
print("Mean Absolute Error =\t" + str(mean_absolute_error(y_test, y_pred)))
print("Mean Squared Error =\t" + str(mean_squared_error(y_test, y_pred)))
print("Root Mean Square =\t" + str(np.sqrt(mean_squared_error(y_test, y_pred))))