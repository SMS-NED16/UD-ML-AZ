# Hierarchical Clustering

## Introduction
- Like all clustering algorithms, HC classifies unlabelled data into clusters, which are often similar and sometimes identical to that of K-means.
- The end result is similar to that of K-means but the procedure is different.
- Two types of clustering
	1. Agglomerative - the focus in this course
	2. Divisive

## Agglomerative HC for a Single Cluster
1. Make each data point into a single point cluster. If there are `N` data points, there will be `N` clusters.
2. Take the two closest data points and make them one cluster. This forms `N-1` clusters.
3. Take the two closest clusters and make them one one cluster. This forms `N-2` clusters.
4. Repeat step 3 until there are is only a single cluster remaining.
5. Finish algorithm

## Distance between Clusters
- Can use Euclidean distance when we're dealing with individual vectors.
- But what about distance between clusters?
- Closeness of clusters isn't as straightforward as a Euclidean distance between the two.
- We could
	1. Take the distance between the two closest points.
	2. Take the distance between the two furthest points.
	3. Take the average of the distance between all possible combinations of points in two clusters.
	4. Take the distance between centroids of the clusters.
- Based on our particular problem, we can choose a specific definition for the distance between clusters.

## Building Dendrograms
- The HC algorith maintains a memory of the steps it took to combine clusters in a structure called the **dendrogram**.
- We begin with each individual point as an individual cluster. 
- We then find the two closest points P2 and P3 and combine them into a single cluster.
	- On the dendrogram, we want to signify that P2 and P3 were combined together into a cluster. We draw a bar between the P2/P3 marks on the dendrogram that is proportional to their Euclidean distance.
	- The dendrogram height is determined by the distance between them, and the distance measures the "dissimilarity" between the two. 
	- The more similar two data points are, the closer they will be to each other on the scatter plot. So this makes sense.
	- The dissimilarity is captured by the height of the bars in the dendrogram. 
- The next two closest clusters are P5 and P6. We draw another bar between P5 and P6 on the dendrogram, and since these two clusters are further apart than P2 and P3, the height of the bar is higher to signify the higher dissimilarity.
- The next closest clusters are the one formed by P2/P3 and P1. We draw a bar on the dendrogram that goes from P1 to the **midpoint of P2/P3 bar**. 
- Similar bar between P4 and midpoint of P5/P6.
- Final bar represents final cluster between P456 and P123 and goes from midpoint of P13 to midpoint of P4.
- Following the heights of the bars can help us track the order in which clusters were formed: shorter bars, closer clusters.

## Using Dendrograms
- We follow the heights of the dendrogram in ascending order to identify the order in which clusters were formed.
- We set dissimilarity thresholds/horizontal thresholds and can define that we don't want dissimilarity to exceed this threshold.
- Within a cluster, we don't want a dissimilarity greater than this threshold.
- The threshold prevents formation of any clusters with greater dissimilarity.
- Can quickly tell how many clusters there will be for a given threshold by checking the number of vertical lines that cross the threshold: the number of lines that cross the threshold = the number of clusters in the model.
- Suppose we reduced the threshold for our example to just below the level of dissimilarity that would let us combine P1 with P23 and P4 with P56.
- This means we will have 4 clusters: P1, P23, P4, P56.
- If the dissmilarity threshold was even lower, we'd have even more clusters.
- **We can use the dendrogram to help us decide the optimal number of clusters**.
	- Extend all horizontal lines along the extent of the dendrogram.
	- Find the longest vertical line that doesn't cross any of these horizontal lines.
	- This line represents the **largest distance**.
	- Recommended approach: take a threshold that will cross that largest distance and then use that threshold to calculate the number of clusters (using lines that cross the new horizontal line).


