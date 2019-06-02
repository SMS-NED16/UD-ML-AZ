#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 06:33:43 2019

@author: saadmashkoor

kernel_pca.py - Using a kernel trick to extract principal components for a social
network advertising dataset for classification using logistic regression. 
"""

"""--------------------------------PREPROCESSING------------------------------"""
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# extract features and labels
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

"""--------------------------------KERNEL PCA---------------------------------"""
# Import the KernelPCA class and instantiate an object with a Gaussian Kernel
from sklearn.decomposition import KernelPCA
kernel_pca = KernelPCA(n_components=2, kernel='rbf')
X_train_kpca = kernel_pca.fit_transform(X_train)
X_test_kpca = kernel_pca.transform(X_test)


"""------------------------------LOG REG MODEL---------------------------------"""
from sklearn.linear_model import LogisticRegression
logreg_classifier = LogisticRegression(random_state=0)
logreg_classifier.fit(X_train_kpca, y_train)
y_pred = logreg_classifier.predict(X_test_kpca)

"""--------------------------------METRICS------------------------------------"""
from sklearn.metrics import classification_report, confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

"""-------------------------------VIZUALIZE------------------------------------"""
from matplotlib.colors import ListedColormap

X_set, y_set = X_test_kpca, y_test

# Domain is a grid of features from the min to max value of each separated by 0.01
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01))

# assume each 0.01 division in grid is a user, make prediction for it, color it 
plt.contourf(X1, X2, 
             logreg_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha=0.75, cmap=ListedColormap(('red', 'green')))

# Set limits for each axis
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# for every unique class in the target set
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label=j)

# Annotate
plt.title('Wine Segmentation - KPCA (Test)')
plt.xlabel('KPC1'); plt.ylabel('KPC2'); plt.legend(); plt.show()
