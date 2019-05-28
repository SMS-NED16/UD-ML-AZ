#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:17:30 2019

@author: saadmashkoor
"""

"""k_means_clustering.py - Unsupervised K-Means Clustering for mall customers."""

"""
Dataset contains ID, age, salary, and spending score for 200 customers of a mall's
membership card database. Spending score is computed by the mall for each client to
assess how much the client spends: the closer it is to 100, the more the client
spends. The mall has asked us to use `Annual Income` and `Spending Score` to
try and segment its users and identify subgroups in their customers: classic clustering problem
"""

#%reset -f

# import libraries
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Mall_Customers.csv')

# extract features we're interested in - no labels because unsupervised clustering
X = dataset.iloc[:, [3, 4]].values

# using the Elbow Method to find the optimal number of clusters
from sklearn.cluster import KMeans

# Find the WCSS for 10 different values of clusters
WCSS = []                           # empty list to store WCSS for each cluster value
for i in range(1, 11):
    # Create an object of the k-means class with the current number of clusters
    # Avoid random initialization trap by user k-means++ for init
    # n_init is the number of times the algo will be run
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, 
                    n_init=10, random_state=0)

    # fit the model to the data
    kmeans.fit(X)

    # compute WCSS for the current number of clusters - called inertia
    WCSS.append(kmeans.inertia_)

# Plot the elbow method graph to identify optimal cluster count
plt.figure(); plt.plot(range(1, 11), WCSS); plt.xlabel('K - Clusters');
plt.ylabel('WCSS - Inertia'); plt.title('Elbow Method - WCSS Against No. of Clusters');
plt.grid(True); plt.show(); plt.tight_layout()

# From inspection, we can tell that the optimal number of clusters is 5
k_ideal = 5
kmeans_ideal = KMeans(n_clusters=k_ideal, init='k-means++', max_iter=300,
                      n_init=10, random_state=0)

# Fit to the data X and predict which cluster it belonds to
y_k_means = kmeans_ideal.fit_predict(X)

# Visualize the clustering results (only for 2D data)
plt.figure(); 

# This line of code finds the x and y coordinates of all data points in X
# where the predicted class is 0
plt.scatter(X[y_k_means == 0 , 0], X[y_k_means == 0, 1], s=100, c='red', label='Careful')

# Repeat for all other classes
plt.scatter(X[y_k_means == 1 , 0], X[y_k_means == 1, 1], s=100, c='blue', label='Standard')
plt.scatter(X[y_k_means == 2 , 0], X[y_k_means == 2, 1], s=100, c='green', label='Target')
plt.scatter(X[y_k_means == 3 , 0], X[y_k_means == 3, 1], s=100, c='cyan', label='Careless')
plt.scatter(X[y_k_means == 4 , 0], X[y_k_means == 4, 1], s=100, c='magenta', label='Sensible')

# Plot the centroids
plt.scatter(kmeans_ideal.cluster_centers_[:, 0],
            kmeans_ideal.cluster_centers_[: ,1],
            s=300, c='yellow', label='Centroids')

plt.title("Clusters of Clients"); plt.xlabel('Annual Income - $1000s');
plt.ylabel("Spending Score (1-100)"); plt.legend(); plt.grid(True)