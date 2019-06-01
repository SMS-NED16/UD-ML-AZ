# How NNs Learn

NNs aren't explicitly programmed to perform a specific task. While we do structure/construct a specific NN architecture, the NN **learns** how to perform its task on its own by adjusting its parameters to minimise the output of its **cost function**.

## Perceptron - Single Row
- The perceptron is the simplest kind of neural network that can "learn" how to perform a task or "train" itself.
- First proposed in 1957 (!!!)
- A single layer neural network consisting of inputs `X1, X2, X3,..., Xm`, a single hidden layer with one neuron, and a **predicted** output value `y'`. 
- `y'` is the predicted output, while `y` is the actual output for this set of input featurs.
- The hidden layer uses the weighted sum of inputs with an activation function to produce an output value.
- We compare `y'` with `y` and quantize their difference using the **cost function**.
- **Cost Function** = `C = 0.5 * (y' - y)^2`.
- There are many other choices of cost function, but this one in particular simplifies the process of gradient descent.
- The cost function produces an output that is proportional to the error in the prediction.
- The cost function output is then passed **back** to the hidden layer and is used to update the weights `W1`, `W2,`..., `W_m`. 
- We feed these three values into the NN with the updated weights and compare the new prediction with the actual output. 
- Once again, the cost function output is propagated back through the neural network and is used to adjust the weights.
- This iteration of input -> NN -> Cost function -> Cost -> NN -> adjust weights is repeated until we achieve a sufficiently low difference/error between `y'` and `y`.
- It is important to remember that every time we're feeding in exactly the same values for features into the perceptron. 

## Multiple Rows
- **One Epoch** is when we pass all the rows of the dataset into the neural network to produce an output.
- Once we've passed all rows of the dataset to the NN and measured their predicted outputs, we can find the sum of their individual costs.
- Once we have the full cost function, we update the weights for the neural network.
- **These weights are the same for all rows. We don't compute new weights for each row.**
- Repeat the epoch as many times as necessary to achieve a sufficiently low cost function output.