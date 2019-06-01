# Backpropagation

- **Forward propagation**: data is entered into the input layer, passes through the layers, and is used to compute the error using a cost function.
- **Backpropagation**: The cost function's output is passed back along the neural network from O/P to I/P layer that allows all weights of the neural network to be adjusted simultaneously to minimise cost function output using gradient descent.
- The advantage of backprop is that all weights update simultaneously.

# ANN Workflow Summary
1. Randomly initialize all weights of the neural networks using values that are close to 0 (but not exactly 0.)
2. Input the first observation of your dataset in the input layer, with each feature forming one input node.
3. Forward Propagation: from left to right, the neurons are activated in a way that the impact of each neuron's activation is limited by the weights. Propagate these activations until getting the predicted result y. 
4. Compare the predicted result othe actual result. Measure the generated error.
5. Back Propagation: from right to left, the error is back-propagated. Update the weights according to how much they are responsible for the error. The learning rate decides by how much we update the weights. 
6. Repats steps 1 - 5 and update the weights after each observation (reinforcement learning). Or, repeat steps 1- 5 but update the weights only after a batch of observations (batch learning).
7. When the whole training set has passed through the ANN, an epoch is complete. Redo more epochs if necessary. 