#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 08:41:51 2019

@author: saadmashkoor
"""

"""logistic_regression.py - Implementing Logistic Regression for classification
of a Social Network ad data. We want to use existing clickstream data to predict
if a new social media user is likely to purchase an SUV."""

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# extracting features
# dataset contains a Male/Female column (categorical) - LabelEncoding 
# For simplicity, we'll drop this column and avoid having to deal with encoding
# Also because we will be able to visualize results 
X = dataset.iloc[:, [2,3]].values # only Age and Salary used as predictors
y = dataset.iloc[:, -1].values

# Large dataset so will use train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling - we want accurate predictions - no need to scale y
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)    
X_test = sc_X.transform(X_test)

# Instantiate and train logistic regression model
from sklearn.linear_model import LogisticRegression
log_classifier = LogisticRegression(random_state=0)
log_classifier.fit(X_train, y_train)

# Make predictions using the logistic regression model
y_pred = log_classifier.predict(X_test)    # predictions of 1/0 on each observation

# Evaluate Performance - Confusion Matrix & Classification Report functions
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)


"""Logistic regression will always create a linear decision boundary. The 
decision boundary divides in the input space into two regions, one corresponding
to each class (1/0). The ideal classifier would create regions such that there
would be no observations of the other class in that region - such classifiers are
usually non-linear, and are very difficult to build. There will always be some
False Positives and False Negatives. The linear boundary that we will visualize
next is the best that the logistic regression model could do with the given data.

However, this is still useful because it tell us that most buyers were older and
had higher salaries, and most people who did not buy the SUV despite clicking on
the ad were younger with lower salaries.

The boundary is defined by the training set - not the test set."""

# Visualizing Training Set Results
# ListedColormap class helps colorize data points
from matplotlib.colors import ListedColormap

# Local variables for plotting
X_set, y_set = X_train, y_train

# Domain is a grid of features from the min to max value of each separated by 0.01
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01))

# assume each 0.01 division in grid is a user, make prediction for it, color it red/green for 0/1
# contour makes a dividing line between the two regions
plt.contourf(X1, X2, log_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha=0.75, cmap=ListedColormap(('red', 'green')))

# Set limits for each axis
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# for every unique class in the target set (i = 0, j = 1)
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label=j)
plt.title('Logistic Regression (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualizing Test Set Results
plt.figure()
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01))
plt.contourf(X1, X2, log_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set ==j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label=j)
plt.title('Logistic Regression (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()