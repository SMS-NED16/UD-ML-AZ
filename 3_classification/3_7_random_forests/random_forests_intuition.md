# Random Forests for Classification

## Ensemble Learning
- Techniques in which multiple ML algorithms are combepsined together to create a larger, more complex ML algo.
- Can be the same ML algo or different algos.
- Random Forest is an ensemble method: it aggregates the results of several decision trees for classification.

## Steps in Random Forest Classification
1. Pick `K` data points from the dataset.
2. Build a decision tree for these `K` data points.
3. Choose the number `NTree` of trees you want to build and repeat steps 1 and 2 for each tree.
4. For a new data point, make each of your `NTree` trees predic the category to which the data point belongs, and assign the new data point to the category that wins the majority vote.

Each tree is built on a randomly selected subset of the data, and even though an individual tree may not be the best at prediction, leveraging the power of several trees helps us make more accurate predictions.

MSFT used random forests for building Kinect. 