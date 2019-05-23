#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 18:25:13 2019

@author: saadmashkoor
"""

"""Revising the Python implementation of Polynomial Regression"""
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Small datset so no train-test split required

# Only one feature with relatively same range so no scaling required

# Fitting Linear Regression model to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizing the Linear Regression results
plt.figure(); plt.scatter(X, y, color='red', label='Actual')
plt.plot(X, lin_reg.predict(X), color='blue', label='Linear Model')
plt.title('Salary vs Employee Level - Linear Model'); plt.grid(True); plt.legend();
plt.xlabel('Employee Level'); plt.ylabel('Salary ($)');

# Visualizing the 2nd degree Polynomial Regression results
plt.figure(); plt.scatter(X, y, color='red', label='Actual');
plt.plot(X, lin_reg_2.predict(X_poly), color='blue', label='Linear Model')
plt.title('Salary vs Employee Level - Degree 2'); plt.grid(True); plt.legend();
plt.xlabel('Employee Level'); plt.ylabel('Salary ($)');

# Visualizing the 3rd degree Polynomial Regression results
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_poly, y)

plt.figure(); plt.scatter(X, y, color='red', label='Actual');
plt.plot(X, lin_reg_3.predict(X_poly), color='blue', label='Linear Model')
plt.title('Salary vs Employee Level - Degree 3'); plt.grid(True); plt.legend();
plt.xlabel('Employee Level'); plt.ylabel('Salary ($)');

# Visualizing the 4th degree Polynomial Regression results
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_4 = LinearRegression()
lin_reg_4.fit(X_poly, y)

plt.figure(); plt.scatter(X, y, color='red', label='Actual');
plt.plot(X, lin_reg_4.predict(X_poly), color='blue', label='Linear Model')
plt.title('Salary vs Employee Level - Degree 4'); plt.grid(True); plt.legend();
plt.xlabel('Employee Level'); plt.ylabel('Salary ($)');

# Increasing resolution of domain for 4th degree polynomial
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualize Scatterplot
plt.figure()
plt.scatter(X, y, color='red'); plt.grid(True);
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff [Linear Regression]')
plt.xlabel('Position Level'); plt.ylabel('Salary'); 

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.figure();
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.xlabel('Position Level'); plt.grid(True); plt.ylabel('Salary');
plt.title('Truth or Bluff? [Poly Regression - 4, high res]');


# Truth or bluff? Predict the previous salary of an employee with level 6.5
# Predict with linear regression model
y_pred_lin = lin_reg.predict(6.5)

# Predict with 4th degree polynomial 
y_pred_poly = lin_reg_2.predict(poly_reg.fit_transform(6.5))

# Print the salaries
print("With linear regression:\t" + str(y_pred_lin))
print("With 4th degree polynomial:\t" + str(y_pred_poly))   # 2k below employee salary