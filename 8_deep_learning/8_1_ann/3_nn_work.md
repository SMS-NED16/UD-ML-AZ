# How do Neural Networks Work?

We'll see a neural network in action in the context of a housing price prediction problem. We're not going to train the network in this example. We'll only see how the neural network can take a set of input features and use them to predict the price of a house.

## Input Layer
- We have four input parameters
	- `X1` - area in square feet
	- `X2` - bedrooms
	- `X3` - distance to city in miles
	- `X4` - age in years
 - These four features will make up our input layer.

## Output Layer
- The predicted price of the house with a particular set of features.
- For the simplest neural network, `y` would be the result of an activation function being applied to the weighted sum of all features.
- `y = f(w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4)`
- Even without the hidden layer, we have a representation that works for other ML models.

## Hidden Layer
- Hidden layers add further complexity and predicting power to our neural network.
- Assume we add a hidden layer of 5 nodes in the hidden layer.
- Assume each node in the layer is connected via a synapse to the top node in the hidden layer.
- Not all synapses will have a non-zero weight
	 - X1 and X3 have non-zero weight for the synapse linking them to the top node in layer 2.
	 - X2 and X4 have zero weight.
	 - This neuron probably looks for training samples that have a large area compared to the distance of the house from the city, depending on the weights.
	 - This is why the neuron doesn't really care about the number of bedrooms and age of the house. 
- Similarly, the second neuron has non-zero weights for X1, X2, and X4.
	- This neuron is not concerned with the distance of the house from the city.
	- Maybe there are a lot of families in this city that are looking for new, large properties with lots of bedrooms. 
	- Based on this training data, the neural network has tailored this node to specifically select newer, larger houses with more bedrooms. 
- Similarly, the last neuron in the layer only uses age to produce a prediction.
	- If a property is old, it will obviously require more maintenance, and will not be preferable to customers, which will drive the price down.
	- However, if it is an antique house (100+ years), it may be deemed as historic and its price will increase.
	- This neuron will fire up if the house is above a certain age. 
	- An excellent candidate for the rectifier function: won't fire up until a certain age (the age at which a house officially becomes historic and starts being valuable.)
- Other nodes could be picking up associations that we aren't even aware of.
- This is why hidden layers are important. They add an additional level of complexity to the model to transform the input features into a different set of features that are more useful for making predictions. 