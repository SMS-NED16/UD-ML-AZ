# Softmax and Cross-Entropy

## Softmax
- A kind of layer that we can create in an artificial neural network.
- The output of a softmax layer is always a set of probabilities that add up to 1.
- Technically, the nodes of the output layer of a neural network are not interconnected, so there is no way for one node to know the probability output by another node and to constrain their cumulative probabilities to 1.
- This is fixed using the **Softmax Function** `f_j(z) = e^(zj)/sum(k)[e^zk]`.
- Normally, the predicted outputs `z1`, `z2` would be any arbitrary values, but with the application of the softmax function, the `k` dimensional vector of output values is "squashed" down b/w 0 and 1 so that their sum never exceeds 1.
- The softmax function comes hand in hand with a concept called cross entropy.

## Cross-Entropy
- Mathematical Formulae
	- `L_i = -log(e^{f_yi}/sum[j]e^{f_j})`
	- `H(p, q) = - sum[x][p(x)log q(x)]` for binary classification
		- p is the actual label for a given training example.
		- q is the predicted probability for this label as determined by the NN.
- Intuitively, just as the MSE function is used to compute the "error" between the predicted and actual output of an ANN, the cross-entropy function is a suitable loss function for CNNs with a softmax layer.
- We want to minimise the cross-entropy to maximise the prediction accuracy of a CNN with softmax. 