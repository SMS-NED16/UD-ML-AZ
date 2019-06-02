# Linear Discriminant Analysis (LDA) 

- Similar to principal components analysis but not the same.
- It is also used a dimensionality reduction technique in the pre-processing step for pattern classification.
- Like PCA, its goal is to project the features onto a lower dimension space.
- However, in addition to finding the components that maximize variance, LDA also attempts to find the component axes that **maximize the separation between multiple classes**.
- So there is another constraint in the use of LDA for classification.
- LDA's goal is to project a feature space (a dataset of `n`-dimensional samples) onto a small subspace `k`k (where `k` < `n-1`) while maintaining the class-discriminatory information.
- Both PCA and LDA are linear transformation technqiues used for dimensionality reduction, but while PCA is unsupervised, **LDA is supervised because of its relation to the dependent variable**.

## Steps 
1. Compute the `d` dimensional mean vectors for the different classes from the dataset.
2. Compute the scatter matrices (in-between-class and within-class scatter matrix).
3. Compute the eigenvectors (`e1`, `e2`, `e3`, ..., `ed`) and corresponding eigenvalues.
4. Sort the eigenvectors by decreasing eigenvalues and choose `k` eigenvectors with the largest eigenvalues to form a `d x k` dimensional matrix `W` (where every column represents an eigenvector).
5. Use this `d x k` eigenvector matrix to transform the samples onto the new subspace. This can be summarized by the matrix multiplication `Y = X x W` where `X` is an `n x d` dimensional matrix representing the `n` samples, and `y` is a matrix of the transformed `n x k` dimensional samples in the new subspace.