import numpy as np

class NeuralNetwork(object):
    def __init__(self):
            return
    def sigmoid(self, x):
        return np.exp(x) / (1 + np.exp(x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x)*(1- self.sigmoid(x))

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
        delta_out = self.sigmoid_derivative(z_s[-1]) * (y - a_s[-1])

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
