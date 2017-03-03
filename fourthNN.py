
import numpy as np

class Layer(object):

	def __init__(self, numInput, numOutput):
		self.weights = np.random.randn(numInput, numOutput)
		self.gradient = np.zeros(numFeatures, numOutput)
		self.output = []

	def forward(self, layer_input, weights):
		return np.dot(layer_input, weights)

	def backprop(self):
		pass

	def update(self, gradient):
		pass


class NeuralNetwork(object):

	def __init__(self, layers, X, y, numIterations):
		self.layers = layers
		self.X = X
		self.y = y
		self.numIterations = numIterations

	def train(self):
		for num in self.numIterations:
			for index in range(len(self.X)):
				layer_input = X[index, :]
				
				# Forward Propagation
				for layer in self.layers:
					layer.output = layer.forward(layer_input, layer.weights)
					layer_input = layer.output
				
				# Error for the last layer
				error = self.Y[index, :] - layer.output
				
				# Backpropagation
				layer_input = error
				for layer_num in range(num(layers), -1 , -1):
					layers[layer_num].gradient = np.dot(layer.output, layer_input)

					layer_input = layer.delta

				# Update the weights
				for layer in layers:
					layer.weights -= alpha * layer.gradient
					layer.gradient = np.zeros((numFeatures, numFeatures))

layer1 = Layer(2,2)
layer2 = Layer(2,1)
X = [[1,1], [2,1], [4,1], [6,1]]
y = [3,2,4,5]
network = NeuralNetwork([layer1, layer2], X, y, 1000)
network.train()




