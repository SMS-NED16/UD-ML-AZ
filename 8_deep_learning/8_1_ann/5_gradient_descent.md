# Gradient Descent

In the previous lecture, we discussed how NNs "learn" by adjusting their parameters/weights to minimise the output of a cost function, which measures the difference between the predicted and actual outputs.

The brute force approach to this problem is try out multiple combinations of weights, record their costs, and identify the combination with the minimal cost. 

This approach may be viable when we're dealing with a perceptron, which has only a handful of features, but not for practical neural networks. As the number of features increases, we have to deal with the **curse of dimensionality**. They are are far too many neurons and far too many weights to optimise.

For instance, if there are 25 weights and want to test out 1000 different combinations, the total combinations that the neural network will have to run for will be 1000^75.

**TaihuLight**, one of the world's fastest supercomputers, operates at 93 peta FLOPS (93 x 10^15 floating-point operations per second). 

For argument's sake, assume it takes only 1 FLOP for TaihuLight to compute the cost for 1 combination of weights. This means, Taihu will take 10^75 / (93 x 10^15) = 1.08 x 10^58 seconds = 3.42 x 10^50 **years** to test all the combinations.

**This is longer than the universe has actually existed (LOL)**. And we're only dealing with a simple neural network. 

So we clearly need to find a faster, more efficient way to compute optimal weight values for a neural network to minimise its cost.

## What Does Gradient Descent Do?
- Compute the **gradient** of the loss function with respect to the weights. Then change weights in the opposite direction using chain rule. 