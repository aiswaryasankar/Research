import numpy as np
import random as random

# First I will write a linear function that will output y values for given x values. This function will be in the form y = ax + b. 
# Then I will compute the error.
# What I need to do now is to update the weights with respect to the gradient of mean function.
# I want a while loop that will continue to iterate until the gradient is smaller than a given value.
# 
# From there I will try to learn the values for each of the functions.  
# Now how do I want to loop through this?
# Let me try one more iteration and check that the values are decreasing for the weights.
# Then I will put this into a formal loop.

class Neuron(object):
	def __init__(self, numFeatures):
		#self.weights = [random.uniform(-.1, .1) for num in range(2)]
		self.weights = [5,6]
		self.features = numFeatures
		self.bias = 1

	def forward(self, X):
		# I am initializing the arrays randomly
		return np.dot(X, self.weights)

	def train(self):
		# Create your data
		a = 1
		b = 2
		X = [[m, 1] for m in range(1, 10)]
		y = [a*m + b for m in range(1, 10)]

		# Now I want to loop through and learn the weights
		gradient = 10.0
		count = 0
		while (count < 50):
			gradient = (2.0/ len(y) * sum(np.dot(np.dot(X, self.weights) - y, X)))
			print ('gradient is ')
			print(gradient)
			print('weight is ')
			for i in range(len(self.weights)):
				self.weights[i] = self.weights[i] - 0.01 * gradient
			print(self.weights)
			count += 1

		# for i in range(200):
		# 	print('iteration ' + str(i))
		# 	y_hat = self.forward(X)
		# 	gradient = self.loss(X, y, y_hat)
		# 	print(self.weights)
		# 	self.weights -= .1 * gradient
		# 	print(self.weights)
		

	def loss(self, X, y, yHat):
		# The loss is the mean of the sum of the differences between the predicted and the actual y values
		print(sum(np.abs(y - yHat)))
		print(1.0/len(y) * sum((y - yHat)**2))
		print('gradient')

		gradient = 2.0/len(y) * (np.dot (np.dot(X, self.weights) - y, X))
		print(gradient)
		return gradient

	def backprop(self):
		print('')

neuron = Neuron(2)
neuron.train()


