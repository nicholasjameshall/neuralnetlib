from layers.layer import Layer

import numpy as np
from scipy import signal

class Convolutional(Layer):
    """
    input_shape: tuple of (depth, height, width) representing the input.
    kernel_dim: int representing size of each kernel matrix.
    num_kernels: int representing how many kernels we want.
    """
    def __init__(self, input_shape, kernel_dim, num_kernels, learning_rate):
        self.learning_rate = learning_rate

        self.input_shape = input_shape
        input_depth, input_height, input_width = input_shape
        self.num_kernels = num_kernels
        self.input_depth = input_depth
        self.output_shape = (
            num_kernels,
            input_height - kernel_dim + 1,
            input_width - kernel_dim + 1
        )
        self.kernels_shape = (
            num_kernels, # number of kernels (e.g. 2)
            input_depth, # how many matrices within the kernel we need (e.g. 1)
            kernel_dim, # size of each matrix (e.g. 2x2)
            kernel_dim
        )

        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input

        # The output will simply be the biases added to the result of the
        # cross-correlation between the input and the respective kernel.
        self.output = np.copy(self.biases)
        # Iterates through the kernals
        # We will want to apply each kernal to the input
        for i in range(self.num_kernels):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(
                    self.input[j], self.kernels[i, j], "valid")
        return self.output
    
    def backward(self, output_gradient):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.num_kernels):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(
                    self.input[j], output_gradient[i], "valid")
                input_gradient[j] = signal.convolve2d(
                    output_gradient[i], self.kernels[i, j], "full"
                )

        self.kernels -= self.learning_rate * kernels_gradient
        self.biases -= self.learning_rate * output_gradient
        return input_gradient