# Kernel SVM for Classification

## SVM Recap
- We had a set of observations which belonged to different classes.
- The goal was to find a decision boundary called the maximum margin that maximised
the separation between the outliers of the classes (support vectors).
- This boundary was **linear**. What happens when the decision boundary for the dataset
is non-linear?

## When to use Kernel SVM?
- When the data is not linearly separable, we can't classify it into different classes
using a hyperplane.
- SVM assumes that data is linearly separable and attempts to find the optimal linear
decision boundary.
- However, when the data isn't linearly separable, SVM can't find a single decision boundary,
let alone the optimal one. 
- This is where the kernel SVM comes in: it is used for **classifying non-linearly separable data**.
- The kernel SVM maps the dataset to a higher dimensional space where the data is linearly separable.
- The kernel trick allows us to do this in a computationally efficient way.

## Mapping to a Higher Dimension 
- We just discussed that the key concept in kernel SVM is to map data from one dimension, in which it is not linearly separable, to a higher dimension where it is linearly separable.
- We can then use SVM in the higher dimension to find a decision boundary and map it back to the original dimension.
- In this way we can use linear decision boundary optimisation methods to derive a non-linear decision boundary. 

## Example 1 - 1D Data
- Consider a one-dimensional binary class dataset with 9 training examples of which 6 belong to class R and 3 to class G.
- The data are not linearly separable: cannot use a single dot to separate the two classes. 
- If we increased this dimensionality of the dataset from 1D to 2D: instead of placing both R and G classes on the x-axis and distinguishing them by their color, we can use a mapping function such as f(x) = x - 5, which will shift all the data to the left.
- The next step is to square the data f'(x) = f(x) * f(x) = (x-5)^2
- This projects the data on a parabola, which makes it linearly separable. 
- This is just one of **many** possible sequences of mapping functions we could have used.
- So mapping 1D data to 2D data made it linearly separable. 
- We can then project the decision boundary back onto the original space and can then use it to functionally separate the data.

## Example 2 - 2D Data
- When 2D  data is arranged in rings (class 1 examples surrounded by a circle of class 2 examples), we once again cannot use a linear classifier.
- We can use a mapping function phi(x1, x2) = (x1, x2, z) to transform this data to 3D space. 
- This will make the data linearly separable in 3D space, where the separator is a hyperplane that separates the two parts of the dataset.
- Once again, we can project it back to 2D space and get a circular decision boundary that separates our classes.

## Problem - Computational Efficiency
- Mapping to a higher dimension can be computationally intensive.
- The larger the dataset, the more computational resources are required for the mapping and projection from higher dimension to lower dimension. 
- This is where the **kernel trick** comes in: it helps us get similar results without necessarily using the computationally intensive mapping to a higher dimension/projection to a lower dimension.

## The Kernel Trick
- Consider the Gaussian or Radial Basis Function
	K(x, l^i) = e^[-||x - l^i||^2/2\sigma^2]
- K stands for `kernel`, and it is a function of `x` (features) and l^i (the ith `landmark` ). 
- The function computes exponent raised to the L2 norm of the difference between the features and the landmark divided by the standard deviation. 
- When our feature vector is far away from the landmark's projection on the domain, the value of the RBF function is very small.
- But when the feature vector is nearer to the landmark's projection on the domain, the value begins to approach e^-0 i.e. 1.
- Optimal placement of landmark is a complex topic that is beyond the scope of this course. The important point to remember is that the landmark is always projected onto the domain.
- Any points within the 'circumfrence' of the Gaussian function's projection on the domain are assigned a positive non-zero value. All others are assigned value of zero.
- Sigma defines how wide the circumfrence of the rbf function will be. The smaller the sigma, the smaller the circumfrence, and the smaller the subset of the domain that will be assigned values > 0.
- **This is the essence of the kernel trick: we have created a decision boundary without going into a higher-dimensional space**.
- We've bypassed all the computations that would have been required to map all our data points to a higher dimensional space. 
- It is also possible to make linear combinations of two different kernel functions with different landmarks to calculate a more complex decision boundary. 

## Types of Kernel Functions
- The Gaussian/**R**adial **B**asis **F**unction (RBF) is not the only kernel function that can be used for classification. 
- Other kernel functions include
	- Sigmoid function K(X, Y) = tanh(gamma.X^TY + r) 
	- Polynomial function K(X, Y) = (gamma.X^TY + r)^d, gamma > 0