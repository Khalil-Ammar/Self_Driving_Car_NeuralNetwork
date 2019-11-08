#References:
# 1. http://neuralnetworksanddeeplearning.com/chap1.html?fbclid=IwAR0NUN7qaHHndp1bhm3fKGkz9DdHvZ-Dobg_UFPY3nU7i1IgW11WiB3TErs
# 2. https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

import numpy as np
import time

def sigmoid(x):
	return np.exp(x) / (1 + np.exp(x))

def sigmoidPrime(self, x):
	return self.sigmoid(x) * (1 - self.sigmoid(x))
		
class NeuralNetwork:
	# This class creates a basic fully-connected neural network.
	# The input to the class constructor is the topology of the network: a list of layers containing the number of neurons in that layer.
	# Example: NeuralNetwork([3, 4, 2]) creates a network with 3 neurons in the first layer, 4 in the second and 2 in the final layer.
	
	def __init__(self, topology):
		self.numLayers = len(topology)
		self.layers = topology
		self.neurons = [np.zeros(x) for x in self.layers[0:]] #create list of lists corresponding to the value of each neuron
		self.weights = [np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])] # create list of 2d lists corresponding to all weights
		self.biases = [np.random.randn(x, 1) for x in self.layers[1:]] #create list of lists corresponding to the bias of each neuron

	def feedForward(self, input):
		self.neurons[0] = input
		for i in range(1, len(self.layers)):
			for j in range(0, len(self.neurons[i])):
				self.neurons[i][j] = sigmoid(np.dot(self.neurons[i - 1], self.weights[i - 1][j]) + self.biases[i - 1][j])
		return self.neurons[-1]

    def compute_activations(self, x, W, B):
        a = x
        a_s = [x]
        z_s = []
        for w, b in zip(W,B):
            z = np.dot(w, a)+b
            z_s.append(z)
            a_s.append(sigmoid(z))

        return (z_s, a_s)

    def backpropagation(self, x, y, W, B):
        z_s, a_s = self.compute_activations(x, W, B)
        delta_b = [np.zeros(b.shape) for b in B]
        delta_w = [np.zeros(w.shape) for w in W]

        ##compute error in output neuron
        delta_out = self.sigmoidPrime(z_s[-1]) * (y - a_s[-1])

        ##initialize last element of delta_b and delta_w
        delta_b[-1] = delta_out
        delta_w[-1] = np.dot(delta_out, a_s[-1].transpose())

        ##backpropagation
        num_layers = len(self.layers)
        delta = delta_out
        for layer in xrange(2, num_layers):
            delta =  np.dot(W[-layer + 1].transpose(), delta) * self.derivative_activation(z_s[-layer])
            delta_b[-layer] = delta
            delta_w[-layer] = np.dot(delta, a_s[-layer - 1].transpose())

        return (delta_b, delta_w)
