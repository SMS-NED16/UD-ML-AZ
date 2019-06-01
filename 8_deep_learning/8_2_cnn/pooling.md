# Max Pooling

## What is Pooling?
- We want our CNN to be able recognize a cheetah as a cheetah, regardless of the size/dimensions/orientation of the image itself or the position/orientation of the cheetah within the image.
- There are lots of little differences b/w images: texture/lighting/orientation/etc.
- This means if the CNN were to look at very minute features such as an exact number of spots in the exact same location and orientation, it will never find them.
- We want the CNN to have **spatial invariance**: even if the feature is a little distorted, our CNN should still be able to find it. 
- This is what pooling does.
- Aka **downsampling**.

## Max Pooling
- Take a box of (n x n) pixels, place it on top of the feature map, and find the maximum value of the section of the feature map within the box. 
- Store the maximum value in a **pooled feature map**.
- Move box by stride, repeat procedure, and store the maximum value again.
- We were still able to preserve the features: the maximum numbers represent max similarity.
- At the same time, we're getting rid of redundant features: since we're taking the maximum value of a subset of the feature map, we're remembering the specific feature that is represented by this value.
- The maximum value makes our feature map more robust: we're still preserving the features while still accounting for their potential spatial distortions.
- We're also reducing the size of the feature map, which will help us improve processing speed and also decrease the number of parameters that will be used in the final layer of the neural network.
- Removing the extra parameters is also important because it prevents overfitting. 

## Why Max Pooling?
- See Evaluation of Pooling Operations in Conv Architectures for Object Recognition by Dominik Scherer et. al (2010).