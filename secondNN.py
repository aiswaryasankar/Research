
	# Create a layer class with forward and backpropagation, initialization
	# Neural Network should only define the structure
	# Train will loop over all the layers
	# Neural network will take in all the layers, each layer will have a matrix of values and weights and then perform the functions
	# Update will be separate to update the weights


class Layer(object):

	def __init__(self, weights):
		#self.weights = [random.uniform(-.1, .1) for num in range(2)]
		self.weights = weights
		self.bias = 1

	def forward(self, X):
		# I am initializing the arrays randomly
		return np.dot(X, self.weights)

	def backprop(self):
		while (count < 50):
			gradient = (2.0/ len(y) * sum(np.dot(np.dot(X, self.weights) - y, X)))
			self.update(gradient)

	def update(self, gradient):
		for i in range(len(self.weights)):
			self.weights[i] = self.weights[i] - 0.01 * gradient


class NeuralNetwork(object):

	def __init__(self, weights):
		#self.layers = 
		self.layers = Layer(weights)
		self.a, self.b = 1, 2
		self.X = [[m, 1] for m in range(1, 10)]
		self.Y = [a*m + b for m in range(1, 10)]
		#for layer in weights:
		#	self.layers.append(Layer(weights))

	def createData(self):
		a, b, c, d = 1,2,3,4
		X = [[m, 1] for m in range(1, 10)]
		y = [c * (a*m + b) + d for m in range(1, 10)]

	def train(self):
		# I want to iterate across all the layers
		for layer in self.layers:
			# For the given layer I want to do a forward iteration and update the X value accordingly
			self.X = layer.forward(self.X)
		# After I do all the forward propagation, I need to do backpropagation
		# For the first layer I have the actual values to subtract from y and x

		# For the second layer I don't have actual y values

		
		# Now I want to loop through and learn the weights
		gradient = 10.0
		count = 0
		while (count < 50):
			# Backprop method
			gradient = (2.0/ len(y) * sum(np.dot(np.dot(X, self.weights) - y, X)))
			print ('gradient is ')
			print(gradient)
			print('weight is ')
			# Update method
			for i in range(len(self.weights)):
				self.weights[i] = self.weights[i] - 0.01 * gradient
			print(self.weights)
			count += 1

#neuron = Neuron(2)
#neuron.train()
# weights = [[4, 5], [6,7]]
weights = [5,6]
NeuralNetwork(weights)


