#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 04:51:10 2019

@author: saadmashkoor

pca_wine.csv - Performing principal components analysis on UCI Wine dataset. 
It contains information about the chemical composition of different wines and the
customer segments that prefer to purchase them. Our job is to make a classification
model to predict which customer segment a wine should be recommended to based
on its chemical composition. We also want to use PCA to extract 2 features that
explain the most variance in the dataset, and to visualize the customer segments.
"""

"""--------------------------------PREPROCESSING------------------------------"""
# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Wine.csv')

# extract features and labels
X = dataset.iloc[:,:-1].values              # all except label column
y = dataset.iloc[:, -1].values              # market segment

# train/test split - 0.20 test_size because only 178 entries, so not a lot of samples
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling - always done for PCA to prevent outliers from skewing results
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

"""-----------------------------------PCA-------------------------------------"""
# import PCA class from `decomposition` module
from sklearn.decomposition import PCA

# instantiate with `None` for n_components because we don't know if only 2
# components will explain enough variance. Need to decide based on cumulative variance.
pca = PCA(n_components=None)

# extract training set independent variables based on variance
X_train = pca.fit_transform(X_train)

# because pca object already fit to features, just transform test set features
X_test = pca.transform(X_test)

# Check cumulatve explained variance - sum of %age of variance explained by each PC
explained_variance = pca.explained_variance_ratio_

# Top 2 PCs explain 36.9% + 19.3% = 56.2% of variance in dataset, which is good!

"""-------------------------------PCA TOP 2-----------------------------------"""
# Since we have confirmed that 2 independent variables can explain sufficiently
# high variance, we create a new PCA object to extract ONLY these features
pca_2 = PCA(n_components=2)
X_train_2 = pca_2.fit_transform(X_train)
X_test_2 = pca_2.transform(X_test)

"""----------------------------TESTING PC MODEL---------------------------------"""
# Train a logreg model using only the features identified by PCA
from sklearn.linear_model import LogisticRegression
logreg_classifier = LogisticRegression(random_state=0)
logreg_classifier.fit(X_train_2, y_train)

# Make predictions
y_pred = logreg_classifier.predict(X_test_2)

# Performance metrics
from sklearn.metrics import confusion_matrix, classification_report
conf_mat = confusion_matrix(y_test, y_pred)                 # excellent results
class_report = classification_report(y_test, y_pred)        # almost no incorrect predictions - 97%

"""---------------------------VISUALIZING RESULTS------------------------------"""
# ListedColormap class helps colorize data points
from matplotlib.colors import ListedColormap

# Local variables for plotting
X_set, y_set = X_test_2, y_test

# Domain is a grid of features from the min to max value of each separated by 0.01
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01))

# assume each 0.01 division in grid is a user, make prediction for it, color it 
plt.contourf(X1, X2, 
             logreg_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha=0.75, cmap=ListedColormap(('red', 'green', 'blue')))

# Set limits for each axis
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# for every unique class in the target set
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label=j)

# Annotate
plt.title('Wine Segmentation - PCA (Test)')
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend(); plt.show()