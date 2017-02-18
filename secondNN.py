
	# Create a layer class with forward and backpropagation, initialization
	# Neural Network should only define the structure
	# Train will loop over all the layers
	# Neural network will take in all the layers, each layer will have a matrix of values and weights and then perform the functions
	# Update will be separate to update the weights
import numpy as np

class Layer(object):

	def __init__(self, weights):
		#self.weights = [random.uniform(-.1, .1) for num in range(2)]
		self.weights = weights
		self.bias = 1

	def forward(self, X):
		# I am initializing the arrays randomly
		return np.dot(X, self.weights)

	def backprop(self, pred, actual):
		gradient = (2.0 / len(pred) * sum(np.dot(np.dot(pred - actual, pred), pred)))
		return gradient

	def update(self, gradient):
		for i in range(len(self.weights)):
			self.weights[i] = self.weights[i] - 0.0001 * gradient
		return self.weights


class NeuralNetwork(object):

	def __init__(self, weights):
		#self.layers = 
		self.layer1 = Layer(weights[0])
		self.layer2 = Layer(weights[1])
		a = weights[0][0]
		b = weights[0][1]
		c = weights[1][0]
		d = weights[1][1]
		self.X = np.array([[m, 1] for m in range(1, 10)])
		self.Y = np.array([6 * (4*m + 5) + 7 for m in range(1, 10)])
		#for layer in weights:

	def train(self):
		# I want to iterate across all the layers
		for layer in self.layers:
			# For the given layer I want to do a forward iteration and update the X value accordingly
			self.X = layer.forward(self.X)
		gradient = 10.0
		count = 0
		while (count < 50):
			# Backprop method
			gradient = (2.0/ len(y) * sum(np.dot(np.dot(X, self.weights) - y, X)))
			# Update method
			for i in range(len(self.weights)):
				self.weights[i] = self.weights[i] - 0.01 * gradient
			print(self.weights)
			count += 1

#neuron = Neuron(2)
#neuron.train()
# weights = [[4, 5], [6,7]]

# Here I want to initialize a Neural Net with 2 layers. The first layer will have the given weights. We will perform forward propagation on the X and the first layer weights.  Then that result with the second layer weights.  I will then compare these results with the actual Y values.
weights = np.array([[4, 5], [6,7]])
neuralNet = NeuralNetwork(weights)
train_weights = np.array([[3,5], [4,8]])
count = 0
while count < 50:
	X_prime = neuralNet.layer1.forward(neuralNet.X)
# I need to make the X matrix into the same dimensions as before
	X_2 = np.array([[val, 1] for val in X_prime])
	X_pred = neuralNet.layer2.forward(X_2)

# Now I need to do backpropagation
# The first layer is going to be the same.  I will subtract the actual values from the predicted values.
# What will my method take in?  It should take in the predicted values and the generated values.  These can be the weights computed or the actual Y values
# Backprop thus takes in two parameters, predction and actual

	diff = neuralNet.layer2.backprop(X_pred, neuralNet.Y)
	updated_weights = neuralNet.layer2.update(diff)
	# This has updated the last layer's weights.  Now I will do the first layer's weights.
	diff2 = neuralNet.layer1.backprop(updated_weights, neuralNet.layer1.weights)
	updated_weights = neuralNet.layer1.update(diff2)
	count += 1

