#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 07:03:54 2019

@author: saadmashkoor
"""

"""svr_rpt.py - Revision of support vector machines in Python"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Extract features and target
X = dataset.iloc[:, 1:2].values             # features
y = dataset.iloc[:, 2].values               # target

# Scaling both X and y since SVR estimator from sklearn does not do this by default
from sklearn.preprocessing import StandardScaler    # import Scaler
sc_X = StandardScaler(); sc_y = StandardScaler();   # Instantiate one for features and target
X = sc_X.fit_transform(X)                           # scale the features
y = sc_y.fit_transform(y.reshape(-1, 1))            # scale the target - reshape to array

# Import and instantiate an SVR model
from sklearn.svm import SVR
svr_regressor = SVR(kernel='rbf')                   # rbf best for non-linear data

# Fit the model
svr_regressor.fit(X, y)

# Visualize results
plt.figure(); plt.scatter(X, y, color='red', label='Actual');
plt.plot(X, svr_regressor.predict(X), color='blue', label='SVR')
plt.xlabel('Employee Level'); plt.ylabel('Salary'); plt.grid(True); plt.legend();
plt.title('Salary vs Employee Level (Scaled SVR Prediction)')


# Undo scaling for more interpretability
X_org = dataset['Level'].values                     # original feature values
y_org = dataset['Salary'].values                    # original target values
X_grid = np.arange(min(X), max(X), 0.1).reshape(-1, 1)
y_pred = svr_regressor.predict(X_grid)              # scaled predictions
X_grid_org = sc_X.inverse_transform(X_grid)         # reverse domain scaling
y_pred_org = sc_y.inverse_transform(y_pred)         # reverse target scaling
plt.figure(); plt.scatter(X_org, y_org, color='red', label='Actual')
plt.plot(X_grid_org, y_pred_org, color='blue', label='SVR')
plt.xlabel('Employee Level'); plt.ylabel('Salary'); plt.grid(True); plt.legend()
plt.title('Salary vs. Employee Level (SVR - Original Scale)');


# Print expected salary of 6.5 level employee
scaled_x = sc_X.transform([[6.5]])                  # scale test value of feature
predicted_y = svr_regressor.predict(scaled_x)       # predict using regressor
actual_y = sc_y.inverse_transform(predicted_y)      # invert scaling for prediction
print("Predicted salary for EL 6.5 = $" + str(actual_y)) # 170.37k