import csv
import numpy as np
import random
from scipy import signal



'''
def preprocess_data(data):
    random.shuffle(data)
    data = np.array(data)
    x = data[:, 1:] # pixel values
    y = data[:, 0].astype(int)  # Labels
    m = x.shape[0]

    # reshape x because convolutional takes a 3d shape with a collection of m images
    x = x.reshape(m, 1, 500, 500) 
    x = x.astype("float32") / 255.0
    # one hot encode y

    # at m, y set the value to one --> EX 1 == ([0,1], ...)
    y_encoded = np.zeros((m, y.max()+1))
    y_encoded[np.arange(m), y] = 1
    y_encoded = y_encoded.reshape(m, 2, 1)
    return x, y_encoded
'''
class Layer():
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass

# depth is the amount of kerneles for each image
class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):

        # |input dim| -->  
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        # |input dim| -->

        # Kernel output size calc --> (output of kernel mult is following eqn)
        # stride size is 1 as seen with the addition
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)

        # initialize kernels and biases
        limit = np.sqrt(2 / (input_depth * kernel_size * kernel_size))
        self.kernels = np.random.randn(*self.kernels_shape) *limit
        self.biases = np.random.randn(*self.output_shape) *.1

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)

        # every kernel for every image layer

        for i in range(self.depth):
            for j in range(self.input_depth):
                # add each ouput of the correlate of input and kernels
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output
        
    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        # for all image input layers (i) and for each kernel per layer (j)
        # calculate the kernel and bias gradients
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                # "full" --> full operator
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")
        
        # update values based on gradients
        # return input gradients to be used as output gradients for the previous layer
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient


#    |||||| reshape for one hot encoded form ||||||
# reshapes output shape to input shape and input to ouput
# Why --> ouptut of the convolutionsal layer is 3d block and the end we use dense layer which takes column vectors as inputs for one hot encoded
class Reshape(Layer): 
    def __init__ (self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def forward(self, input):
        return np.reshape(input, self.output_shape)
    
    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)
    

class Softmax():
    def forward(self, x):
        e_x = np.exp(x-np.max(x))
        self.output = e_x / e_x.sum(axis=0, keepdims=True)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        return output_gradient

#  ||||| Binary Cross Entroy |||||

def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred / np.size(y_true))

def catagorical_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1-1e-12)
    return -np.sum(y_true * np.log(y_pred))

def catagorical_cross_entropy_prime(y_true, y_pred):
    return y_true - y_pred

#  ||||| Activation Layer |||||

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))


#  ||||| Sigmoid Activation |||||  

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            x = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x))
        
        def sigmoid_prime(x):
            s = sigmoid(x)
            x = np.clip(x, -500, 500)
            return s * (1 - s)
        
        super().__init__(sigmoid, sigmoid_prime)


#  ||||| Layer that calculates the neurons in a layer |||||

class Dense():
    def __init__(self, input_size, output_size):
        self.weights = np.random.uniform(-1/np.sqrt(10), 1/np.sqrt(10),(output_size, input_size))
        self.bias = np.zeros((output_size, 1))

    def forward(self, input):
        # input size: (n--> number of pixels)
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    

    def backward(self, output_gradient, learning_rate):

        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient



# softmax gives a probability from 0 to 1 relative to the size of the other data



class MaxPooling(Layer):
    def __init__(self, size, stride):
        self.size = size
        self.stride = stride

    def forward(self, input):
        self.input = input
        n_c, h_prev, w_prev = input.shape

        h = (h_prev - self.size) // self.stride + 1
        w = (w_prev - self.size) // self.stride + 1
        output = np.zeros((n_c, h, w))

        self.max_indices = np.zeros_like(input, dtype=bool)

        for c in range(n_c):
            for i in range(h):
                for j in range(w):
                    h_start = i * self.stride
                    h_end = h_start + self.size
                    w_start = j * self.stride
                    w_end = w_start + self.size

                    region = input[c, h_start:h_end, w_start:w_end]
                    max_val = np.max(region)
                    output[c, i, j] = max_val

                    # Store mask for backprop
                    max_mask = (region == max_val)
                    self.max_indices[c, h_start:h_end, w_start:w_end] = max_mask

        return output

    def backward(self, output_gradient, learning_rate):
        n_c, h_out, w_out = output_gradient.shape
        input_gradient = np.zeros_like(self.input)

        for c in range(n_c):
            for i in range(h_out):
                for j in range(w_out):
                    h_start = i * self.stride
                    h_end = h_start + self.size
                    w_start = j * self.stride
                    w_end = w_start + self.size

                    mask = self.max_indices[c, h_start:h_end, w_start:w_end]
                    input_gradient[c, h_start:h_end, w_start:w_end] += (
                        output_gradient[c, i, j] * mask
                    )

        return input_gradient



