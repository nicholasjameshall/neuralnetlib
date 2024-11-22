import numpy as np

from activation_layers.sigmoid import Sigmoid
from layers.reshape import Reshape
from layers.convolutional import Convolutional
from layers.dense import Dense
from loss.bce import binary_cross_entropy
from loss.bce import binary_cross_entropy_prime
from keras.datasets import fashion_mnist as mnist
from keras.utils import to_categorical

def preprocess_data(x, y):
    # Creates a list of 1x28x28 matrices.
    x = x.reshape(len(x), 1, 28, 28)
    print(x)
    x = x.astype("float32") / 255
    # Creates one-hot encoding.
    # In this case, since we're dealing with a binary classification, this would
    # simply be a list of [1,0] and [0,1] (representing 0 and 1, respectively)
    y = to_categorical(y)
    # Splits the encodings into an extra dimension.
    # FROM: [[0. 1.],...]
    # TO:   [[[0.], [1.]], ...]
    y = y.reshape(len(y), 2, 1)
    return x, y

def filter_data(images, labels, filters, limit=10):
    # Gets lists of the indexes of images in the training set with the required
    # label.
    indices = [np.where(labels == x)[0][:limit] for x in filters]
    # Merges into one list of indices.
    all_indices = np.hstack(indices)
    # Puts them in a random order.
    all_indices = np.random.permutation(all_indices)
    # Returns the ordered lists of images and labels that we care about
    return images[all_indices], labels[all_indices]

def fetch_data(training_samples, test_samples):
    # x = the images, as a 3d array (an ordered list of matrices)
    # y = the labels, as a 1d array (an ordered list of categories)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = filter_data(x_train, y_train, [0, 1], 10)
    x_test, y_test = filter_data(x_test, y_test, [0, 1], 100)
    return preprocess_data(x_train, y_train), preprocess_data(x_test, y_test)

def train(inputs, labels):
    error = 0
    for x, y in zip(inputs, labels):
        output = x
        # Forward
        for layer in network:
            output = layer.forward(output)

        # Calculate error
        error += binary_cross_entropy(y, output)

        # Backward
        grad = binary_cross_entropy_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad)
    
    error /= len(x_train)
    return error

def test(inputs, labels):
    attempts = []
    for x, y in zip(inputs, labels):
        output = x
        for layer in network:
            output = layer.forward(output)

        attempts.append(np.argmax(output) == np.argmax(y))
    
    return np.mean(attempts)

# Define network
learning_rate = 0.1
network = [
    Convolutional(
        (1, 28, 28), # Size of our input. In this case, one single 28x28 image
        3, # How large the kernel should be - 3x3 (in this case)
        5, # How many kernels / feature maps do we want?
        learning_rate
    ),
    Sigmoid(),
    Reshape(
        # Input shape: one output per kernel (5) as defined in conv. layer.
        # The resulting shape is 26x26 because the filter is 3x3.
        (5, 26, 26),
        (5 * 26 * 26, 1) # Output shape: all values squeezed into a column
    ),
    Dense(
        5 * 26 * 26, # Input size, a column
        100, # Output size: 100 (arbitrary)
        learning_rate
    ),
    Sigmoid(),
    Dense(
        100, # Input size: same as previous output
        2, # Output size: arbitrary
        learning_rate
    ),
    Sigmoid()
]

(x_train, y_train), (x_test, y_test) = fetch_data(50, 100)

# Train the model
epochs = 50
for e in range(epochs):
    error = train(x_train, y_train)
    print(f"{e + 1}/{epochs}, error={error}")

# Test it on unseen images
test_result = test(x_test, y_test)
print(test_result)