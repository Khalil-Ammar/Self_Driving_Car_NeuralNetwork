#References:
# 1. http://neuralnetworksanddeeplearning.com/chap1.html?fbclid=IwAR0NUN7qaHHndp1bhm3fKGkz9DdHvZ-Dobg_UFPY3nU7i1IgW11WiB3TErs
# 2. https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

import numpy as np
import time
import pickle

def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))

def sigmoidPrime(x):
	return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
	# This class creates a basic fully-connected neural network.
	# The input to the class constructor is the topology of the network: a list of layers containing the number of neurons in that layer.
	# Example: NeuralNetwork([3, 4, 2]) creates a network with 3 neurons in the first layer, 4 in the second and 2 in the final layer.

    def __init__(self, topology):
        self.numLayers = len(topology)
        self.layers = topology
        self.out_layer_len = topology[-1]	##number of neurons in the last layer
        self.neurons = [np.zeros(x) for x in self.layers[0:]] #create list of lists corresponding to the value of each neuron
        self.weights = [np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])] # create list of 2d lists corresponding to all weights
        self.biases = [np.random.randn(x) for x in self.layers[1:]] #create list of lists corresponding to the bias of each neuron
        self.biases = [b.reshape(b.shape[0],1) for b in self.biases]

	def load_data(self, dataset_path):
        train_data = pickle.load(open(dataset_path+'/train.pkl', 'rb'))
        valid_data = pickle.load(open(dataset_path+'/valid.pkl', 'rb'))

        return train_data, valid_data

	## returns a vector of length 'out_len' where the index y is 1 and everything else is 0
    def vectorize(self, y, out_len):
        arr = np.zeros((out_len,1))
        arr[y] = 1.0
        return arr

	## formats data so it's more convenient to use in our NN algorithm
    def data_wrapper(self, dataset_path, input_shape, out_len):
        train_data, valid_data = self.load_data(dataset_path)

        ## train_data is a tuple(x,y) with x being the input and y a vectorized output
        training_inputs = [row[0] for row in train_data]
        training_results = [self.vectorize(row[1], out_len) for row in train_data]
        train_data = [(x,y) for x,y in zip(training_inputs, training_results)]

        valid_inputs = [row[0] for row in valid_data]
        valid_results = [row[1] for row in valid_data]
        valid_data = [(x,y) for x,y in zip(valid_inputs, valid_results)]

        return train_data, valid_data


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
        a = a.reshape((len(a), 1))
        a_s = [a]
        z_s = [None]
        for i in range(0, len(W)):
            n1, _ = W[i].shape
            z_s.append(np.zeros((n1, 1)))
            a_s.append(np.zeros((n1, 1)))
        for i in range(len(W)):
            z_s[i+1] = np.dot(W[i], a_s[i]) - B[i]
            a_s[i+1] = sigmoid(z_s[i+1])

        return (z_s, a_s)

    def backpropagation(self, x, y):
        W = self.weights
        B = self.biases
        (z_s, a_s) = self.compute_activations(x)
        d_b = [np.zeros(b.shape) for b in B]
        d_w = [np.zeros(w.shape) for w in W]

        ##compute error in output neuron
        delta_out = sigmoidPrime(z_s[-1]) * (y - a_s[-1])

        ##initialize last element of d_b and d_w
        delta = [None]*len(W)
        delta[-1] = delta_out
        d_b[-1] = -delta_out
        d_w[-1] = np.dot(delta[-1], a_s[-2].transpose())

        ##backpropagation
        for i in range(len(W)-2, -1, -1):
            delta_i = np.dot(W[i+1].transpose(), delta[i+1]) * sigmoidPrime(z_s[i+1])
            delta[i] = delta_i
            d_b[i] = -delta[i]
            d_w[i] = np.dot(delta[i], a_s[i].transpose())
        return d_w, d_b

    def calc_error(self, X):
        n_true = 0.0
        for row in X:
            x = row[0]
            t = row[1]
            z, a = self.compute_activations(x)
            if (np.argmax(a[-1]) - t) == 0:
                n_true += 1
        return n_true / len(X)

    def update_weights(self, d_w, d_b, learning_rate):
        self.weights = [w + learning_rate * dw for w,dw in zip(self.weights, d_w)]
        self.biases = [b + learning_rate * db for b,db in zip(self.biases, d_b)]

    def SGD(self, train_data, valid_data, max_epoch, learning_rate):
        ##print initial loss and accuracy
        valid_acc = self.calc_error(valid_data)
        print("Evaluate using randomly initialized weights:\tValidation Acc: {0:4.4f}\n\n".format(valid_acc))
        ## perform gradient descent
        for e in range(max_epoch):
            for row in train_data:
                ##extract input and outputs
                x = row[0]
                y = row[1]

                ## backpropagation and update weights
                d_w, d_b = self.backpropagation(x, y)
                self.update_weights(d_w, d_b, learning_rate)

            ## print accuracy and loss for each epoch
            valid_acc = self.calc_error(valid_data)
            print("Epoch #{0:4}/{1}\tValid Acc: {2:4.4f}".format(
                e, max_epoch, valid_acc))

        return

if __name__ == '__main__':
    ## init network
    input_shape = (160*120,1)
    out_layer_length = 4
    nn = NeuralNetwork([input_shape[0], 100, 200, out_layer_length])

    ##set parameters
    learning_rate = 0.1
    max_epoch = 100

    ##get data
    train_data, valid_data = nn.data_wrapper("./dataset/vision",input_shape, out_layer_length)


    ##train network
    nn.SGD(train_data, valid_data, max_epoch, learning_rate)
