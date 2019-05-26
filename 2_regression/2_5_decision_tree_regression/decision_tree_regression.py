#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 06:22:50 2019

@author: saadmashkoor
"""

"""decision_tree_regression.py - A Python program to predict the salary
of an employee based on employee level using Decision Tree Regression"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Extracting features and target variable
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Building the first tree w/o feature scaling

# Train/Test Split - not necessary because a very small dataset

# Instantiating Decision Tree Regression model
from sklearn.tree import DecisionTreeRegressor    # Import

# First tree is made with default parameters, random_state set to 0
dtree_regressor = DecisionTreeRegressor(random_state=0)

# Fit the regressor to the training data
dtree_regressor.fit(X, y)

# Make predictions with the regressor 
y_pred_dtree_1 =  dtree_regressor.predict(X)

# Visualize predictions relative to actual data
plt.figure(); plt.scatter(X, y, color='red', label='Actual');
plt.plot(X, y_pred_dtree_1, color='blue', label='Predicted (Default Tree)');
plt.grid(True); plt.xlabel('Employee Level'); plt.ylabel('Salary'); plt.legend();
plt.title('Salary vs Employee Level - Default Decision Tree');

"""The decision tree predictions seem to follow the training data perfectly.
This is not a good thing - it could mean overfitting! The decision tree should
predict an average value that is constant for all data points in a given
leaf. This means the value should be constant for a given interval. In our
figure, we can see the prediction is varying continuously. Either the tree
is considering infinitely many intervals, or there is a problem with the 
way we are plotting predictions. This is the first non-continuous, non-linear model."""

# Predicting value at 6.5
y_pred_6_5 = dtree_regressor.predict(np.array([[6.5]]))    

"""Regressor predicts the salary of a 6.5 level employee is 150k. For the 
first time, we're getting a prediction that is substantially below the 
actual expected value of 160k."""

# Try making predictions for a more continuously varying domain - right now
# we're predicting values for data points only. We need to find predictions
# for actual non-linear, non-continuous model.


# Visualize model results in higher resolution
plt.figure(); plt.scatter(X, y, color='red', label='Actual')
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.plot(X_grid, dtree_regressor.predict(X_grid), color='blue', label='Predicted');
plt.xlabel('Employee Level'); plt.ylabel('Salary'); plt.grid(True);
plt.title('Salary vs Employee Level - High Resolution Decision Tree')


"""Now the decision tree prediction avisualization makes much more sense.
The decision tree is assigning the same average value of salary to each
employee level in the same interval. It has divided the domain into
clearly defined domains (0.5 - 1.5, 1.5 - 2.5, and so on)"""


# Predicted value for employee level 6.5 is the same 
print(dtree_regressor.predict(np.array([[6.5]])))