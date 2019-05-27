#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 10:10:26 2019

@author: saadmashkoor
"""

"""classification_template.py - A template for performing classification tasks
with an arbitrary algorithm using Python"""

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import datased
ds_file_name = 'Data.csv'
dataset = pd.read_csv(ds_file_name)

# Extract features and target
X = dataset.iloc[:, :2].values
y = dataset.ilo[:, -1].values

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling - only for features because target is 0/1
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Instantiate and train a classification model


# Train the model
classifier.fit(X, y)

# Test the model using a confusion matrix and classification report
from skearn.metrics import classification_report, confusion_matrix
c_mat = confusion_matrix(y_test, y_pred)
c_rep = classification_report(y_test, y_pred)


# Define function for visualization
def visualize_results(X, y, classifier):
    # ListedColormap class helps colorize data points
    from matplotlib.colors import ListedColormap
    
    # Local variables for plotting
    X_set, y_set = X, y
    
    # Domain is a grid of features from the min to max value of each separated by 0.01
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01))
    
    # assume each 0.01 division in grid is a user, make prediction for it, color it red/green for 0/1
    # contour makes a dividing line between the two regions
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
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
    
# Call the visualization function for training set
visualize_results(X_train, y_train, classifier)

# Call the visualization function for test set
visualize_results(X_test, y_test, classifier)