# Convolution

## Definition
- A combined integration of two functions.
- Describes how one signal modifies the shape of another signal. 
- (f * g)(t) = int[-infty, infty)]f(tau) x g(t-tau) dtau
- Jianxin Wu's paper **Introduction to Convolutional Neural Networks** is a great introduction to the mathematics behind CNNs.

## Intuition
- A feature detector is a matrix is (usually) a 3 x 3 matrix.
- Also called a **kernel** or **filter**. 
- Intuitively, we take a subset of the input image (stored as a pixel array) of the same size as the feature detector.
- We then compute the sum of all element-wise multiplications between the kernel and the original image. 
- When corresponding elements in the kernel and input subset will be identical, the feature map will store a non-zero sum.
- The **stride** is the amount by which the feature detector "shifts" along the input image.
- The result of this operation is a **Feature Map**, and is also called a **convolved feature** or **activation map**.

## What does convolution do?
- **Size Reduction**: The larger the stride, the smaller the resulting feature map, and the faster the image processing algorithm. This **is not** a lossless process - some information is indeed lost. 
- **Feature Extraction**: But the point is to identify specific features, not to retain every single detail of an image. 
- We create multiple feature maps to obtain our first convolutional layer. The filter applied to extract each feature map can be different.
- This is why **feature detector** is a better term than filter: each matrix extracts or accentuates specific features in the original image in the form of a feature map.
- Such feature detectors are commonly known as image filters in photo editing: sharpen, blur, edge detect, sepia, invert, edge enhance, emboss, etc.
- Different feature detectors (filters) get different feature maps.
- The CNN will automatically learn the appropriate feature detectors needed to extract a specific feature map at each layer. 

## Key Takeaways
- Convolution helps us find features in an image by passing the image through a specific feature detector.
- The result is a feature map, which still preserves the spatial relations between individual pixels.
- The feature maps extracted by a CNN will usually be meaningless/difficult to intepret for humans, but that does not mean they are useless. 