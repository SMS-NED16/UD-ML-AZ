# Activation Function

- The activation function is a mathematical operation used by each node/neuron in a neural network to produce an output based on the weighted sum of its inputs.
- Several different kinds, but each maps the weighred sum (independent variable `z`) to an output between 0 and 1.

## Threshold Function
- f(z) = 0 if x < 0, and f(z) = 1 if x >= 0 
- A yes/no function, very rigid.
- Basically a unit step.

## Sigmoid Function
- f(z) = 1/(1 + exp(-z)) where 
- Also the function used in logistic regression.
- Unlike the threshold function, this is a very smooth function.
- Gradually progresses from 0 to 1.
- f(z) = 0.5 at z = 0.
- Very useful in the output layer of neural networks that are used for classification problems.

## Rectifier
- Not exactly smooth, but it is continuous unlike the threshold.
- f(z) = max(z, 0)
- This basically means that f(z) = 0 for all values of z <= 0, but rises linearly for all values of z > 0.
- Despite having a 'kink', it is one of the most popular activation functions in neural networks.

## Hyperbolic Tangent
- Very similar to the sigmoid function.
- But unlike the sigmoid function, its output values are in the range [-1, 1].

Refer to Xavier Glorot's paper for more information on why Rectifier function is so useful/popular in deep learning.


## Exercises
- **If the dependent variable is binary (y = 0 or 1), which activation function to use?**
	- Threshold function. It only produces 0 or 1 as an output. No values in between.
	- Sigmoid function. Also produces values between 0 and 1, but the output varies smoothly. We can use the sigmoid function to predict the **probability** of y = 1 `P(y = 1) = f(z)`.
- Common workflow
	- Rectifier function in hidden layers.
	- Sigmoid function in the output layer.