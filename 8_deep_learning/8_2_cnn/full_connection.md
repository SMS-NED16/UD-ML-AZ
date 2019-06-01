# Full Connection

## Chaining ANNs to CNNs
- In this step, we add an entire ANN to the CNN.
- Convolution -> Pooling -> Flattening -> ANN Input Layer -> Fully Connected ANN Hidden Layer(s) -> ANN Output Layer.
- In ANNs, hidden layers don't **have** to be fully connected, but in CNNs this is a must.
- The ANN combines features into more features/attributes that better predict the output classes.
- We could technically use our max pooled feature maps to make predictions, and it would work.
- But we also know that ANNs are specifically designed for extracting/combining existing features/attributes to improve the accuracy of a prediction. Why not leverage it?
- The only difference to this ANN will be that the output layer can contain multiple output nodes: it won't necessarily be a binary classification problem. 
- The ANN will use CNN output data to compute a predicted output class, then use it to compute a loss function, then use the loss function to optimise weights/params of the ANN using SGD and backpropagation.
- The other thing that is adjusted is the **feature detectors** in the CNN. 
- The process repeats until the network is optimised.
- The data goes all the way from an image being fed into a CNN to an ANN to a predicted output class.
- Through several iterations, the nodes in the hidden layers that are associated with a positive result for a specific class will adjust their weights so that they have stronger associations with that output class.