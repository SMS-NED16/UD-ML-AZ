#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:51:33 2019

@author: saadmashkoor
"""

"""hier_clustering.py - Agglomerative Hierarchical Clustering in Python"""

# import libraries
import pandas as pd
import matplotlib.pyplot as plt

# import  mall dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')

# extract features - For simplicity only Annual Income and Spending Score are used
X = dataset.iloc[:, [3, 4]].values

# use a dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch # scipy hierarchical clustering functionality

# create a dendrogram
# linkage is a hierarchical clustering algo that we apply to X
# the method `ward` tries to minimise the variance in each cluster
# instead of minimising WCSS like in KMeans, we're minimising variance in each cluster
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

# plot the dendrogram
plt.title("Dendrogram"); plt.xlabel("Customers");
plt.ylabel("Euclidean Distances/Dissimilarity"); plt.show();

"""By following Kirill's procedure, we're able to find out that the ideal
number of clusters in the dataset is 5."""

# fit the agglo hierarchical clustering algorithm to the data with 5 clusters

# import the AgglomerativeClustering class
from sklearn.cluster import AgglomerativeClustering

# instantiate
#   affinity defines how to measure distance
#   linkage minimised variance in individual clusters
hca = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage='ward')

# fit the hierarchical clustering algorithm to the data X, record all classifications
y_hc = hca.fit_predict(X)

# visualize clustering results
plt.figure()
for i in range(min(y_hc), max(y_hc) + 1):
    plt.scatter(X[y_hc == i, 0], X[y_hc == i, 1], s=100, 
                label='Cluster ' + str(i+1), edgecolor='black')

# annotate
plt.legend(); plt.xlabel('Salary - $1000s'); plt.ylabel('Spending Score');
plt.title('Agglomerative Hierarchical Clustering - Mall Customers');
plt.show()


"""INTERPRETATION
- Cluster 5 have high income and come to the mall often, but don't spend much
money. They're careful with the money they spend.
- Cluster 2 represents customers with average income and spending score. They
are standard customers.
- Customers in cluster 3 have high income and also enjoy spending money
in the mall. They are a potential target of marketing efforts. However,
will also need to compare with standard customers. 
- Cluster 4 contains customers wiht low income and high spending score. They
are careless with the money they spend. 
- Cluster 5 is low income, low spenders. They are sensible customers. 
- We can provide these insights to a marketing team to help inform their
decisions about allocating marketing budget.
"""