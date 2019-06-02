#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 06:06:16 2019

@author: saadmashkoor

lda_wine.py - Wine market segmentation using UCI's Wine dataset with LDA.
"""

"""--------------------------------PREPROCESSING------------------------------"""
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Wine.csv')

# extract features and labels
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

"""----------------------------------LDA--------------------------------------"""
# import the LDA class from sklearn.discriminant analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# instantiate object with 2 components - called Linear Discriminants
# Don't need to build a vector of explained variance b/c goal is to separate
# classes and visualize them. 
lda = LinearDiscriminantAnalysis(n_components=2) 

# fit the LDA object to the training data - supervised, so must also pass labels
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)                  # don't need y test because not fitting

"""---------------------------LOG REG CLASSIFIER------------------------------"""
# import model
from sklearn.linear_model import LogisticRegression

# instantiate model
logreg_classifier = LogisticRegression(random_state=0)

# fit model to LDA training data
logreg_classifier.fit(X_train_lda, y_train)

# make predictions
y_pred = logreg_classifier.predict(X_test_lda)

"""---------------------------PERFORMANCE METRICS------------------------------"""
from sklearn.metrics import confusion_matrix, classification_report
conf_mat = confusion_matrix(y_test, y_pred)                 # 100% accuracy!
class_report = classification_report(y_test, y_pred)        # perfectly separable classes

"""-------------------------------VISUALIZING----------------------------------"""
# ListedColormap class helps colorize data points
from matplotlib.colors import ListedColormap

# Local variables for plotting
X_set, y_set = X_test_lda, y_test

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
plt.title('Wine Segmentation - LDA (Train)')
plt.xlabel('LD1'); plt.ylabel('LD2'); plt.legend(); plt.show()
