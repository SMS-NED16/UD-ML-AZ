# Support Vector Machines (SVM) for Classification


## Problem Definition
- SVMs were developed in 1960s, refined in 1990s, and became popular in the 2010s.
- They are somewhat different to other ML algorithms, and are very powerful for both regression and classification.
- Consider a dataset with two features X1 and X2 with all points belonging to one of two preset classes, R and G.
- We want to create a decision boundary between the two clusters of data points that will help us assign a class to a new data point.
- One method is to draw a straight line (horizontal, vertical, infinite diagonals).
- Can create a lot of straight lines as decision boundaries, but each will have different consequences for classification. 
- We want to find the best line/decision boundary for separation.

## Support Vectors and Hyperplanes
- SVM works on the principle of **maximum margin** between data points belonging to different classes.
- The maximum margin is the line that both separates the two clusters of data points **AND** is also equidistant from the closest points in both clusters.
- The points that are equidistant from the the optimum line/maximum margin are called the **support vectors**.
- Only the support vectors, the points closest to the maximum margin, are relevant to the SVM method.
- Each point is called a vector because in higher dimensional space, each data point corresponds to a vector in multidimensional space. 
- The line in the middle is called the **Maximum Margin Hyperplane** or **Maximum Margin Classifier**, hyperplane being a more general term that also applies to higher dimensional data.
- The lines/planes equidistant from the maximum margin hyperplane and tangential to/touching the support vectors are called **Positive and Negative Hyperplanes**.
- It doesn't matter which one is considered positive or negative.
- Any vector of features to the "right" of the positive hyperplane is classified in the positive class, while any vector of features to the "left" of the negative hyperplane is classified in the negative class.

## Why are SVMs Popular?
- Imagine we're trying to teach a machine to differentiate apples from oranges using a dataset of labeled data.
- We then provide an apple/orange to the algorithm as test data.
- Standard ML algos look at the most stock standard kinds of apples and oranges (data points near the center of each cluster), and learns from them instead of the outliers.
- SVMs are different and focus on outliers: they look at apples that are very much like oranges and likewise at oranges that are more like apples. 
- These outliers becomes the **support vectors**: they are very close to the Maximum Margin Hyperplane.
- SVMs look at the extreme cases that are closer to the decision boundary and uses them to make classifications.
- This is why SVMs tend to perform much better than other ML algorithms.	
