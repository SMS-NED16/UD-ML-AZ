# Deep Learning Intuition

## History
- Neural networks and deep learning have been around since the 60s or 70s.
- They became popular in the 80s: lots of research and development funding poured into this field in 1980s.
- However, the trend slowly died off and research funding ceased.
- Why? Technology/hardware was not up to the right standard to facilitate neural networks.
- Requirements for Deep Learning
	- Need a lot of data.
	- Need hardware that can quickly and efficiently process that data.
	- Need hardware to store that data.

## Storage Capacity
- Storage capacity increased 2x between 1956 and 1980, and by 25,600x between 1980 and 2017.
- This increase in size of available memory is an exponential trend.
- The logarithmic cost of hard drive cost per GB is decreasing every year, and is almost 0 now because of free cloud storage options. 


## Introduction to Deep Learning and Neural Networks
- Popularised by Geoffrey Hinton - AI researcher at Google.
- Neural networks are partly inspired by the human brain. 
- A neuron in the human brain can be connected to up to 1000 other neurons through axons/dendrons (outputs/input connections).
- Neural networks are based on the same concept: they're a mathematical construct in which individual nodes or neurons take some input, transform them, and pass them to other neurons at the output.
- Neurons in neural nets are arranged in layers. The first layer accepts input (input layer), processes it/transforms it, and passes it along to a series of **hidden layers** through interconnections. 
- A sequence of hidden layers perform similar transformations by passing its outputs to inputs of the next layer until we reach the output layer, which produces a finite number of outputs (depending on the application).
- The **deep** in deep learning refers to the phenomenon of stacking several such hidden layers between I/O layers. This is called **deep learning** because each additional layer of neurons/hidden layer represents additional depth of the complexity to the transformation performed by the neural network.
