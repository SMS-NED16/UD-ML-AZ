#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:44:10 2019

@author: saadmashkoor
"""

"""k_means_rpt.py - Revising unsupervised learning with K-Means Clustering
algorithm."""

# import libraries
import pandas as pd
import matplotlib.pyplot as plt 

# import the dataset
dataset = pd.read_csv('Mall_Customers.csv')

# extract the features - we're only interested in Salary and Spending Score
X = dataset[['Annual Income (k$)', 'Spending Score (1-100)']].values

# use the elbow method to identify optimum clusters
from sklearn.cluster import KMeans
wcss = []
for i in range (1, 11):
    # Instantiate a K-Means Clustering estimator with the current number of loop counter
    k_means_i = KMeans(n_clusters=i, init='k-means++', n_init=10, 
                       max_iter=300, random_state=0)
    
    # Fit the estimator to the training data
    k_means_i.fit(X)
    
    # Compute Within Cluster Sum of Squares (Inertia)
    wcss.append(k_means_i.inertia_)
    
# Plot WCSS against number of clusters to find optimal clusters by inspection
plt.figure(); plt.plot(list(range(1, 11)), wcss); plt.xlabel('Clusters - K');
plt.ylabel('WCSS - Inertia'); plt.title('Elbow Method - WCSS against Clusters');
plt.grid(True); plt.tight_layout()

# ideal cluster count identified after inspection of elbow graph
cluster_count = 5

# instantiate new K-Means clustering estimator with the new cluster count
kmeans = KMeans(n_clusters=cluster_count, init='k-means++', n_init=10,
                max_iter=300, random_state=0)

# get predicted cluster IDs for each data point
y_kmeans = kmeans.fit_predict(X)

# visualize results
plt.figure()
# plot color-coded data points for each cluster
for i in range(min(y_kmeans), max(y_kmeans) + 1):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, 
                label='Cluster ' + str(i+1), edgecolor='black')

# plot cluster centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300,
            c='yellow', edgecolor='black')

# annotate
plt.legend(); plt.xlabel('Salary - $1000s'); plt.ylabel('Spending Score');
plt.title('K Means Clustering - Mall Customers')