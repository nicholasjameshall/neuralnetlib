from layers.layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size, learning_rate):
        # Initialise the weights and biases to random values
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.learning_rate = learning_rate

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= self.learning_rate * weights_gradient
        self.bias -= self.learning_rate * output_gradient
        return input_gradient