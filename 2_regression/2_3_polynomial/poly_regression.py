#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 06:23:04 2019

@author: saadmashkoor
"""
"""poly_regression.py - an implementation of polynomial regression in Python.
This is the first non-linear regression model of the course."""

"""DATA PREPROCESSING"""
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Position_Salaries.csv')

"""The dataset shows the salaries for different positions in a company.
We need to decide if a new employee demanding 160k is a reasonable request.
There is a non-linear relationship between the position level and the salary.
We will be building a bluffing detector to assess whether or not it is 
possible for an employee with 6 years of experience as Region Manager to
earn 160k."""

# We don't need the `Position` column because the `Level` column provides
# the same information in relatively ordered numerical form.
# So only one feature: Employee Level.
# Our dependent variable will be the salary.

# Extracting features
X = dataset[['Level']].values       # employee level - is a matrix
y = dataset.iloc[:, -1].values      # Salary - is a vector

"""No need for train-test split. Only 10 observations. Not enough observations
possible to train and test separately. We want to make as accurate a prediction
as possible, and this is only possible if we use all the 10 examples for training."""

# No need for Feature Scaling. Only one feature, range is not too wide.
# sklearn applies feature scaling automatically.

"""FITTING A LINEAR REGRESSION MODEL TO THE DATASET"""
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

"""FITTING A POLYNOMIAL REGRESSION MODEL TO THE DATASET"""
# Import PolynomialFeatures library to transform lin features to poly ones
from sklearn.preprocessing import PolynomialFeatures    

# poly_reg feature will be a transform object that will change the 
# matrix of features into a matrix of higher powers of x
# Initially make a poly reg model of degree 2
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X, y)    # return new matrix of features with unity feature for bias

# Create and train a new linear regression object for the poly reg model
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)     # polynomial features, not linear


"""VISUALIZING LINEAR REGRESSION RESULTS"""
plt.figure(); plt.scatter(X, y, color='red', label='Actual');
plt.plot(X, lin_reg.predict(X), color='blue', label='Linear Model');
plt.xlabel('Position'); plt.ylabel('Salary ($)'); plt.grid(True); plt.legend();
plt.title('Salary vs Employee Level (Linear Model)');


"""VISUALIZING POLYNOMIAL REGRESSION RESULTS"""
plt.figure(); plt.scatter(X, y, color='red', label='Actual'); # use the X_poly features!
plt.plot(X, lin_reg_2.predict(X_poly), color='blue', label='Poly Model');
plt.xlabel('Position'); plt.ylabel('Salary ($)'); plt.grid(True); plt.legend();
plt.title('Salary vs Employee Level (Polynomial Model - degree 2))');

"""INTERPRETATION
- The linear model shows that the 160k salary for a level 6 employee is 
actually low. The model suggests that a level 6.5 employee should be offered
300k.
- The linear model is not a great fit to the data. Just by inspection we 
can tell the residual sum of squares will be very large. 
- The polynomial model is a much better fit to the data. Residual sum of
squares is very small.
- According to the polynomial model, the 160k estimate for a level 6.5 employee
sounds about right: the expected salary for such an employee is 180k/200k.
"""


"""ADDING A DEGREE TO THE POLYNOMIAL MODEL TO SEE IF RESULTS IMPROVE"""
# Train
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X, y)
lin_reg_3 = LinearRegression().fit(X_poly, y)

# Test/Visualize
plt.figure(); plt.scatter(X, y, color='red', label='Actual'); # use the X_poly features!
plt.plot(X, lin_reg_3.predict(X_poly), color='blue', label='Poly Model');
plt.xlabel('Position'); plt.ylabel('Salary ($)'); plt.grid(True); plt.legend();
plt.title('Salary vs Employee Level (Polynomial Model - degree 3)');


"""3rd degree polynomial is a better fit. Less deviation between scatterplot
and curve of best fit. Both CEO and 6.5 employees' salaries are close to the
training data. The 160k prediction for 6.5 is almost exactly 160k, so the 
new employee may actually receive brownie points for honesty."""


"""ADDING ANOTHER DEGREE TO THE POLYNOMIAL MODEL TO SEE IF RESULTS IMPROVE"""
# Train
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X, y)
lin_reg_4 = LinearRegression().fit(X_poly, y)

# Test/Visualize
plt.figure(); plt.scatter(X, y, color='red', label='Actual'); # use the X_poly features!
plt.plot(X, lin_reg_4.predict(X_poly), color='blue', label='Poly Model');
plt.xlabel('Position'); plt.ylabel('Salary ($)'); plt.grid(True); plt.legend();
plt.title('Salary vs Employee Level (Polynomial Model - degree 4)');

# I am starting to think we're overfitting the data. W/o a test set we have no way of knowing.


"""IMPROVING RESOLUTION OF CURVE"""
# 90 levels between 1 and 10
X_grid = np.arange(min(X), max(X), 0.1);
X_grid = X_grid.reshape(len(X_grid), 1)
plt.figure(); plt.scatter(X, y, color='red', label='Actual'); # use the X_poly features!
plt.plot(X_grid, lin_reg_4.predict(X_grid), color='blue', label='Poly Model');
plt.xlabel('Position'); plt.ylabel('Salary ($)'); plt.grid(True); plt.legend();
plt.title('Salary vs Employee Level (Polynomial Model - degree 4)');