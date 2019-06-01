# Principal Component Analysis - PCA

## Introduction
- One of the most commonly used dimensionality reduction algorithms.
- Is a **feature extraction** technique as opposed to **feature selection** (backward selection, forward selection, bidirectional selection covered in section 2).
- Used for
	- Visualization
	- Feature extraction
	- Noise Filtering
- Identifies and detects correlation between variables. If two variables are strongly correlated, they are removed from the set of features.
- This maps or projects the features from a higher dimension to a lower dimension while still maintaining the relevance of the data for prediction/classification.
- The intuition is to select the variables/features that are responsible for the greatest variance in the output. 
- This makes PCA an **unsupervised** model: we aren't concerned with the O/P of the dataset.
- PCA can help us reduce our dataset to the two or threee features that help us visualize the dataset.

## Steps
1. Standardize the data.
2. Obtain eigenvectors and eigenvalues from the covariance matrix or correlation matrix, or perform Singular Vector Decomposition.
3. Sort eigenvalues in descending order and choose `k` eigenvectors that correspond to the `k` largest eigenvalues, where `k` is the number of dimensions of the new feature subspace.
4. Construct the projection matrix `W` from the selected `k` eigenvectors.
5. Transform the original dataset `X` via `W` to obtain a `k`-dimensional feature subspace `Y`.

## Summary
- PCA is not about predictions. It is about inference.
- We want to learn about the relationship between X and Y values.
- We then want to find the list of principal axes.
- While it can be skewed by outliers, it is still one of the most commonly used dimensionality reduction techniques.