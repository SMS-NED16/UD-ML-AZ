#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 19:03:56 2019

@author: saadmashkoor
"""

"""regression_template.py - Python program to build an arbitrary regression
model with visualization capabilities for 1D data."""
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')      # change dataset
X = dataset.iloc[:, 1:2].values                     # change indexes  
y = dataset.iloc[:, 2].values

# train-test split if necessary
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
"""

# Perform feature scaling if necessary. Most libs will do this automatically
"""from sklearn.preprocessing import StandarScaler
scaler = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting non-linear regression model to the dataset


# Predict with 4th degree polynomial 
y_pred_poly = regressor.predict(6.5)


# Visualizing non-linear regression model predictions - only if 1D data
plt.figure(); plt.scatter(X, y, color='red', label='Actual');
plt.plot(X, regressor.predict(X), color='blue', label='Linear Model')
plt.title('Salary vs Employee Level - Degree x'); plt.grid(True); plt.legend();
plt.xlabel('Employee Level'); plt.ylabel('Salary ($)');


# Visualizing non-linear regression model with higher resolution and smoother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X), 1))       # reshape to array
plt.figure(); plt.scatter(X, y, color='red', label='Actual');
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Linear Model')
plt.title('Salary vs Employee Level -  Degree x [High Res]'); plt.grid(True); plt.legend();
plt.xlabel('Employee Level'); plt.ylabel('Salary ($)');