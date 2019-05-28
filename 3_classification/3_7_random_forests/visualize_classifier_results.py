#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 05:31:43 2019

@author: saadmashkoor
"""
import matplotlib.pyplot as plt
import numpy as np

def visualize_results(X, y, classifier, plot_title):
    """Generates a contour map with classification boundary for a dataset
    using the specified classifier"""
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
    plt.title(plot_title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()