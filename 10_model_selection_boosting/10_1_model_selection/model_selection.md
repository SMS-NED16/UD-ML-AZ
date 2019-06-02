# Model Selection

In the process of solving a machine learning problem, we often need to identify and select the **best** model for that particular problem. This could involve
1. bias-variance tradeoff and how it varies between training and testing
2. choosing optimal values for hyperparameters of the same kind of model (i.e. the parameters that are not learnt)
3. comparing and contrasting multiple classification/regression/clustering algorithms.

There needs to be an objective, automated way of selecting the model. There are two techniques for this
1. K-Fold Cross Validation
2. Grid Search

We want to improve the model performance of all ML models we have covered. The easiest way to do this is to choose the best/optimal values for a model's hyperparameters i.e. parameters such as `kernel` in the Kernel SVM model. 

## K-Fold Cross Validation
- Train/test split is the **correct** way of evaluating model performance, but not the best one
- Judging the model performance on only one test set is not the best/most reliable way to assess accuracy.
- K-Fold cross validation splits the dataset into `k` folds or sections. The model is trained for `k-1` folds and tested on the `k`th fold. 
- This is repeated for several iterations, with a different combination of testing and test folds for each iteration.
- We then take the average of the accuracies predicted for the test folds of all iterations, and also calculate standard deviation to assess variance. 
- This makes our analysis much more relevant as we will be minimising the variance associated with a single test set.