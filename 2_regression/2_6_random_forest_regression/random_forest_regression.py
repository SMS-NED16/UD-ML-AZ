#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 07:15:00 2019

@author: saadmashkoor
"""

"""random_forest_regression.py - Random Forest Regression to predict employee
salaries"""

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Position_Salaries.csv')

# extracting features and target
X = dataset['Level'].values.reshape(-1, 1)
y = dataset['Salary'].values.reshape(-1, 1)

# Feature scaling not required because sklearn does this by default

# No train/test split required because a very small dataset

# Instantiating, fitting, and training Random Forest regressor
from sklearn.ensemble import RandomForestRegressor

# Define the number of trees in the forest
n_trees = 300;
rf_regressor = RandomForestRegressor(n_estimators=n_trees, random_state=0)
rf_regressor.fit(X, y)

# Predicting and visualizing results
X_grid = np.arange(min(X), max(X), 0.01).reshape(-1, 1)
y_pred = rf_regressor.predict(X_grid)
plt.figure(); plt.scatter(X, y, color='red', label='Actual')
plt.plot(X_grid, y_pred, color='blue', label='Predicted')
plt.grid(True); plt.legend(); plt.xlabel('Employee Level'); plt.ylabel('Salary');
plt.title('Salary vs Employee Level - Random Forest (' + str(n_trees) + ' trees)');

# Prediction for Employee level 6.5
y_pred_6_5 = rf_regressor.predict(np.array([[6.5]]))


"""There will be several trees, each with its own intervals and average 
values for that prediction. There will be more intervals, and therefore 
different values for the average prediction in that interval

A random forest has more steps than with one decision tree. We also have 
a lot of splits of the whole range of levels and a lot more levels as well.
Each level is the average of predicted value from each tree.

However, adding more trees doesn't mean more steps. The more trees we add,
the more the average of the votes will converge to the same value.

Prediction for Employee 6.5
TREES               SALARY
-----------------------------------
10                  167k
100                 158.3k
300                 160.33k 
1000                161.6k
10000
"""