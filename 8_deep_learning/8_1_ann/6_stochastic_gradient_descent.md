# Stochastic Gradient Descent

## Local vs Global Minima in Cost
- It is possible for a neural network's cost function to have multiple local minima, only one of which will be the global minimum.
- Applying gradient descent to a single row of features can result in the cost function's optimizer converging to a **local** rather than global minimum.
- This means the weights computed by the algorithm won't really represent the **optimal** values for maximum accuracy of predictions: the cost could be lower (global minimum).
- The solution to this problem is **stochastic gradient descent**.

## SGD vs Batch GD
- Batch GD is when we pass all rows in the dataset into the neural network, have it compute a cost for each combination of input features, find the average cost, and then optimise the weights using gradient descent.
- Batch GD is thus susceptible to being affected by erroneous data/outliers in the feature set.
- Stochastic GD is different. Individual rows are passed to the data set and the weights are updated for each row. 
- SGD helps us avoid the problem of converging to local rather than global minima. This is because SGD algo has much higher fluctuations - it is analysing one row at a time. 
- It is also faster b/c it doesn't have to load up all the data into memory before it can compute weights.
- Mini-batch gradient descent is a variant of this algorithm that compute weights using batches of rows instead of all rows in the dataset.