# ReLU Layer

## What is the ReLU layer?
- **Re**ctified **L**inear **U**nit.
- Implemented immediately after the convolution layer that extracts feature maps.
- This eliminates linearity in the image.
- We want to increase non-linearity in our image because images themselves are highly non-linear!
- Individual pixels, colours, motifs, edges in the image are all non-linear.
- Without processing the convolutional layer feature maps through a ReLU layer, we risk maintaining linear relationships between pixels and features such as motifs/edges in our feature maps. 
- E.g. ReLU will eliminate white/gray/black progression in pixels in a grayscale image. 
- Very complex mathematics involved in understanding why ReLU and non-linearity are important for the filter output of all intermediate layers in CNNs.
- See CC Jay Kuo's paper and Kaiming He et al. on ImageNet.