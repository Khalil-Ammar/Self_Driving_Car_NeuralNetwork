#References:
# 1. http://neuralnetworksanddeeplearning.com/chap1.html?fbclid=IwAR0NUN7qaHHndp1bhm3fKGkz9DdHvZ-Dobg_UFPY3nU7i1IgW11WiB3TErs
# 2. https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

import numpy as np
import time

def sigmoid(x):
	return np.exp(x) / (1 + np.exp(x))

def sigmoidPrime(x):
	return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
	# This class creates a basic fully-connected neural network.
	# The input to the class constructor is the topology of the network: a list of layers containing the number of neurons in that layer.
	# Example: NeuralNetwork([3, 4, 2]) creates a network with 3 neurons in the first layer, 4 in the second and 2 in the final layer.

    def __init__(self, topology):
        self.numLayers = len(topology)
        self.layers = topology
        self.neurons = [np.zeros(x) for x in self.layers[0:]] #create list of lists corresponding to the value of each neuron
        self.weights = [np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])] # create list of 2d lists corresponding to all weights
        self.biases = [np.random.randn(x) for x in self.layers[1:]] #create list of lists corresponding to the bias of each neuron

    def feedForward(self, input):
        self.neurons[0] = input #place inputs into the first layer of the network
        for i in range(1, len(self.layers)):
        	#iterate over every layer
        	for j in range(0, len(self.neurons[i])):
        		#iterate over every neuron in every layer
        		self.neurons[i][j] = sigmoid(np.dot(self.neurons[i - 1], self.weights[i - 1][j]) + self.biases[i - 1][j]) #compute the values of the jth jth layer of neurons
        return self.neurons[-1] #return the list of the values of the output layer

    def compute_activations(self, x):
        W = self.weights
        B = self.biases
        a = np.copy(x)
        a_s = [a]
        z_s = [None]
        for i in range(0, len(W)):
            n1, _ = W[i].shape
            z_s.append(np.zeros((n1, 1)))
            a_s.append(np.zeros((n1, 1)))

        for i in range(len(W)):
            z_s[i+1] = np.dot(W[i], a_s[i]) + B[i]
            a_s[i+1] = sigmoid(z_s[i+1])

        return (z_s, a_s)

    def backpropagation(self, x, y):
        W = self.weights
        B = self.biases
        (z_s, a_s) = self.compute_activations(x)
        d_b = [np.zeros(b.shape) for b in B]
        d_w = [np.zeros(w.shape) for w in W]

        ##compute error in output neuron
        delta_out = sigmoidPrime(z_s[-1]) * (y - a_s[-1])  ## scalar values

        ##initialize last element of d_b and d_w
        delta = [None]*len(W)
        delta[-1] = delta_out
        d_b[-1] = -delta_out
        d_w[-1] = np.dot(delta[-1][np.newaxis].transpose(), a_s[-2][np.newaxis])

        ##backpropagation
        for i in range(len(W)-2, -1, -1):
            delta_i = np.dot(W[i+1].transpose(), delta[i+1]) * sigmoidPrime(z_s[i+1]) ## eg i = 0 : W[1]' = 10x1, delta[1] = 1x1, z_s[1] = 10x1 ==> result = 10x1
            delta[i] = delta_i
            d_b[i] = -delta[i]
            d_w[i] = np.dot(delta[i][np.newaxis].transpose(), a_s[i][np.newaxis])

        return d_w, d_b, z_s, a_s

nn = NeuralNetwork([5000, 300, 20])
nn.backpropagation(np.random.randn(5000), np.random.randn(20))
