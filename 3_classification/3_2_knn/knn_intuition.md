# K-Nearest Neighbours (KNN) - Intuition

## Introduction
- Consider a scatterplot of two variables, X1 and X2, containing two categories, 1 and 2.
- How do we classify a new data point added to the scatterplot? Should it belong to 1 or 2?
- KNN is classification algorithm. Like logistic regression, it helps us classify a new 
data point or observation on the basis of other examples.
- Unlike logistic regression, it can create non-linear decision boundaries and works on the basis
of examining classes of the K nearest data points instead of all the data points in a subset of 
the domain (such a leaf.)

## Step-by-Step Guide to KNN
- Choose the number of neighbours `K`. One of the most common default values is 5.
- Take the `K` nearest neighbours of the new data point using Euclidean distance or a similar metric.
- Among these `K` neighbours, count the number of data points in each category.
- Assign a new data point to the category where you counted the most neighbours.
- Model is ready. 

## Euclidean Distance
- For two dimensional data (such as our example), the Euclidean distance between two points
$(x_1, y_1)$ and $(x_2, y_2) is measured using $$ d = \sqrt((x_2 - x_1)^2 + (y_2 - y_1)^2)$$
- This can also be extended to $n$ dimensional space i.e. an example with more than 2 features.

## Example
- We compute the distance between the new data point and all training examples, and identify the
the 5 points nearest to the data point (smallest distance).
- We then count the classes of these points: 3 belong to category 1, 2 to category 2.
- Based on this information, we classify the new point as belonging to category 1. 