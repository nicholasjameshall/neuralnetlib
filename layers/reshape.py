from layers.layer import Layer

import numpy as np

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)
    
    def backward(self, output_gradient):
        return np.reshape(output_gradient, self.input_shape)