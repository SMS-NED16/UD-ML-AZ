# CNN Summary

1. We started with an input image to which we applied multiple feature detectors/filters to form feature maps through convolution between subsets of the image and a feature detector matrix.
2. We then applied the ReLU function to the output of step 1 to get rid of linearity in the images. This is the end of the convolutional layer. 
3. We created a pooling layer by storing the max value in a subset of each filter map at the end of the convolutional layer. This makes our CNN **spatially invariant**, reduces size of the feature map, and also prevents overfitting.
4. We flattened all the pooled images into one long vector of values.
5. Flattened CNN output is fed as input of fully connected ANN. The last hidden layer in the network performs **voting** to associate specific features with the specific output classes.
6. Through several epochs of backpropagation, the ANN weights and CNN feature detectors are adjusted using GD to optimise NN ability to recognize images and classify them.