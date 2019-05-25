#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 06:13:43 2019

@author: saadmashkoor
"""

"""svr.py - Support Vector Regression model for a salary dataset."""

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Extracting features and target
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Small dataset - train/test split not needed.

# Will try to build a model without feature scaling

# Fitting SVR to dataset
from sklearn.svm import SVR
svr_regressor = SVR(kernel='rbf')         # rbf is better for non-linear models
svr_regressor.fit(X, y)                   # compute params for SVR

# Predicting a new result
y_pred = svr_regressor.predict(6.5)       # 130k - not a great prediction

# Visualising the SVR results
plt.scatter(X, y, color='red', label='Actual');
plt.plot(X, svr_regressor.predict(X), color='blue')
plt.title('Salary vs Employee Position Level')
plt.xlabel('Position Level'); plt.ylabel('Salary'); plt.legend(); plt.grid(True)

# The SVR model is a horizontal line. This is not a great model at all.


"""The SVR class does not implement feature scaling in its algorithms. Let's
see if feature scaling improves the model."""
# Extracting features and target again
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values.reshape(-1, 1) # the standard scaler expects a 2D array

# Scaling the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler(); X = sc_X.fit_transform(X);     # once for features
sc_y = StandardScaler(); y = sc_y.fit_transform(y);     # then for target

# Create a new SVR model with the scaled features
svr_scaled = SVR(kernel='rbf')
svr_scaled.fit(X, y)

# Predicting a new result
y_pred = svr_scaled.predict(6.5)

# Visualizing scaled SVR results
plt.figure(); plt.scatter(X, y, color='red', label='Actual');
plt.plot(X, svr_scaled.predict(X), color='blue')
plt.title('Salary vs Employee Position Level [Scaled SVR]')
plt.xlabel('Position Level'); plt.ylabel('Salary'); plt.legend(); plt.grid(True)

# The visualization and predictions are both scaled. The last real observation
# point has a large deviation from the SVR model. This is because this data point
# is an outlier for the chosen (default) penalty parameters.


"""CONVERTING SCALED PREDICTION BACK TO ORIGINAL RANGE"""
# We can't just use 6.5 as the argument for our predictor because the predictor
# expects all input features to be scaled. So 6.5 must also be scaled.
# The transform method expects an array - single pair of brackets means a vector, not array
y_pred_scaled = svr_scaled.predict(sc_X.transform(np.array([[6.5]])))

# Now invert the scaling transformation for the scaled y object
y_pred_true = sc_y.inverse_transform(y_pred_scaled)

# Print to console
print("Expected earning for employee at level 6.5" + str(y_pred_true))
# Prediction is 170k, which is very close to the original prediction.


"""VISUALIZING RESULTS FOR HIGH RES DOMAIN"""
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))       
plt.figure(); plt.scatter(X, y, color='red', label='Actual');
plt.plot(X_grid, svr_scaled.predict(X_grid), color='blue', label='SVR')
plt.title('Salary vs Employee Level -  SVR [High Res]'); plt.grid(True); plt.legend();
plt.xlabel('Employee Level'); plt.ylabel('Salary ($)');