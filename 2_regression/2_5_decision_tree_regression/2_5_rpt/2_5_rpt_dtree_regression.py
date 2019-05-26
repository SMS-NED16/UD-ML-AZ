#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 06:50:46 2019

@author: saadmashkoor
"""

"""Revision of Decision Tree Regression with Python for Employee Salary data."""

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Position_Salaries.csv')

# extract features and target
X = dataset['Level'].values.reshape(-1,1)
y = dataset['Salary'].values.reshape(-1,1)

# No need for feature scaling as sklearn's decision tree model does this by default

# No need for train/test split because very small dataset

# Instantiating a Decision Tree regression model
from sklearn.tree import DecisionTreeRegressor
dtree = DecisionTreeRegressor(random_state=0)   # for reproducing results

# Fit the Decision Tree to the training data
dtree.fit(X, y)

# Make prediction for employee level 6.5
y_pred_6_5 = dtree.predict(np.array([[6.5]]))

# Visualize predictions for a high res domain since non-linear, discontinuous predictor
plt.figure(); plt.scatter(X, y, color='red', label='Actual'); plt.grid(True);
X_grid = np.arange(min(X), max(X), 0.01).reshape(-1, 1)
plt.plot(X_grid, dtree.predict(X_grid), color='blue', label='Predicted');
plt.xlabel('Employee Level'); plt.ylabel('Salary'); plt.legend();
plt.title('Employee Salary vs Employee Level - Decision Tree Model')