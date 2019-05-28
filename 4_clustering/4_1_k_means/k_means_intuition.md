# K-Means Clustering

## Clustering
- Similar to classification, but the basis is different. In clustering, we don't lnow what we're looking for.
- A form of unsupervised machine learning in which we try to identify segments or clusters in our data.

## Introduction 
- Consider a scatterplot of data that is **unlabeled**. We had two variables X1 and X2 in the dataset and we plotted them on a set of axes.
- Can we identify clusters or groups among our data? How do we go about doing this?
- K-Means eliminates the complexity in identifying clusters and automatically identifies segments/clusters/sections in the dataset.
- Can be applied to an arbitrary number of variables. 

## How does K-Means Clustering work?
1. Choose the number of clusters. Selecting an optimal number of clusters depends on the context.
2. Select at random K points which will be **centroids** of our clusters. They don't necessarily have to be from our dataset.
3. Assign each data point to the closest centroid. This forms the first set of K clusters. `Closeness` depends on Euclidean/geometrical distances.
4. Compute and place the new centroid of each cluster.
5. Reassign each data point to the new closest centroid. (Repetition of step 3, basically). If any reassignent took place, go to step 4. Otherwise FIN.

## Visualizing the Steps
- **Step 1** For our example, we assume there will be 2 clusters. `K` = 2. (Optimal number). 
- **Step 2** We have to choose centroids for the two clusters. They can be any arbitrary points in our domain.
- **Step 3** For every data point in the domain, compute the distance from centroids R and B, and assign the data point to the closest centroid.
	- A simpler way to do this is to draw a straight line between the two centroids R and B and draw its perpendicular bisector. As the bisector is equidistant from either centroid, any point above the bisector is closer to centroid B. All below the line are assigned to centroid A.
	- **Closest** is an ambiguous term. In mathematics and data science, we can compute other kinds of distances: Euclidean or other kind? 
	- We need to specify the **type of distance** when we set up the algorithm.
	- For simplicity, we will use Euclidean distances.
- **Step 4** Compute and place new centroids of each cluster. Now that we've labeled all data points, we assign a new centroid that is the "center of mass" for the new points. 
- **Step 5** We reassing each data point to the new centroids. If any reassignment of points has taken place, we repeat step 4. 
	- Reassignment does take place: 3 points are closer to R than to B, which they were originally assigned to. 
	- Placed new centroids, draw a line between them, find perpendicular bisector, 1 R reassigned to B.
	- Recompute centroid, 1R reassigned to 1B. 
	- No further reassignments necessary. Equidistant line does not reassign any other points. 
- Now we proceed to the end of the algorithm.

## Random Initialization Trap
- Assume we have a scatterplot of two variables X1 and X2 and choose 3 clusters.
- Because this is a relatively simple clustering problem, we can initialize our random clusters to the centroids closer to the actual centers of the clusters.
- This will make the algorithm converge faster. 
- So even though we could have randomly assigned any value to the cluster, it was useful to choose points that were closer to the 'actual' centroids. This is the benefit of choosing a **good** random initialization.
- What if we changed the centroid location? What would happen if we had a **bad** random initialization?
- Because three clusters, will have to find a point that is equidistant from all three centroids. Then we find lines that are equidistant from a pair of clusters and passes through that point.
- This divides the domain into 3 sections and all data points in that section form a cluster.
- But when we compute the new centroids and perform reassignment, it is possible for nothing to change.
- In this case, we assume the model has converged. 
- But these three clusters are different to the **true** three clusters that we identified by inspection earlier. 
- **So the selection of centroids at the start of the algorithm can dictate the outcome of the algorithm.**
- Solution? Improved K-Means algorithm (aka **K-Means++**)
	- A more involved approach that specifically selects the optimal initial centroids.
	- Always happens in the background with standard ML packages in Pyton and R.
- But still useful to be aware of the random initialization trap and that the centroid initialization can affect the outcome.

## Selecting the Number of Clusters
- How to decide the optimal number of clusters for a given dataset?
- We need a quantifiable metric to evaluate how a certain number of clusters performs to compare and contrast different numbers of clusters.
- **Within Cluster Sum of Squares** - WCSS
- WCSS = Sum of squares of distances between every data point and the centroid in each cluster FOR ALL CLUSTERS.
- So in this case, we'd compute sum of squared distances between all points in clusters 1, 2, 3 an their corresponding centroids.
- If we have performed clustering correctly, all the data points will be closer to their corresponding centroids. The WCSS will be come smaller. 
- The smaller the WCSS, the better the number of clusters chosen for a given problem.
- How far will WCSS keep decreasing?
	- Theoretically, we can have as many clusters as there are examples in our dataset: each point becomes its own cluster/centroid.
	- In this case the WCSS will be 0 because every single point will be the centroid of its own cluster, which means every cluster's sum of squares will be 0.
- WCSS is [0, infty)
- Optimal goodness of fit: ythe WCSS gets better, but is there a cost?
	- At some point, the WCSS decrease per additional cluster becomes infinitesimally small.
	- This is our hint of selecting the optimal number of clusters.
	- We find the **elbow** in our chart where the drop in WCSS goes from being substantial to being incremental.
	- This is the **elbow method** which is used to identify the optimal number of clusters.
- Domain knowledge/prior expertise is used to make judgments about whether or not the elbow truly is the optimal number of clusters.