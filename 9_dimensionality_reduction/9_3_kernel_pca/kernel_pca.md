# Kernel Principal Components Analysis

## Overview
- So far we've covered two dimensionality reduction techniques
	- Principal Components Analysis: identifying features that explain maximum variance in the dataset
	- Linear Discriminant Analysis: identifying features that explain maximum variance **and also** maximise class separation.
- Both of them were used to solve a **linear** classification problem: the wine dataset's segementation results showed that the data could easily be separated by hyperplanes after identifying two components.
- Both techniques only work on linearly separable data.
- What about non-linear problems?

## Kernel PCA Overview
- Kernel PCA is a dimensionality reduction technique used for non-linear data i.e. data that is not linearly separable.
- It maps the features to a higher dimension using a kernel trick and then extracts the principal components in the higher dimensional space.
- It will extract new independent variables when the problem is non-linear.
- Will use this to solve the Social Networks Ad dataset from section 3.