#################################
# Joel Anyanti | 06/20/2018
# Carnegie Mellon University
#################################
# Neural Network Library designed to provide a visual inisght
# to the magic behind deep learning
#################################
# imports
#################################
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

#################################
# Neural Network class
#################################

# define class structure for neural network
class NN:
    def __init__(self, model, X, Y, learning_rate=0.005):
        self.model = model
        self.X, self.Y = X, Y
        self.layers = len(self.model) - 1
        self.params = define_parameters(self.model)
        self.activation_func = None
        self.output_func = None
        self.loss_func = None

    def activation_function(self, activation='relu'):
        self.activation_func = activations[activation]

    def output_function(self, output='sigmoid'):
        self.output_func = activations[activation]

    def loss_function(self, activation='log_loss'):
        self.loss_func = activations[activation]

    def train(self, iterations=100000, cost_interval=100):
        for i in range (iterations):
            self.feed = forward_propagate(self.X, self.params, self.layers, self.activation_func, self.output_func)
            self.cost = calculate_cost(self.Y, self.feed, self.layers, self.loss_func)
            self.grads = backwards_propagate(self.X, self.Y, self.data, self.layers)
            self.params = update_params(self.params, self.grads,self.layers)
            if (i % cost_interval == 0) :
                print(self.cost)
        print(self.cost)
        print(self.params)


#################################
# Activation Functions
#################################

# activation function that returns a value between 0 and 1
def sigmoid(Z):
    return (1/(1+np.exp(-Z)))

# activation function that returns a value between -1 and 1
def tanh(Z):
    return np.tanh(Z)

# activation function that returns the identity of input or 0 if less than 0
def relu(Z):
    return np.maximum(0,Z)

# activation function for multiclass outputs | returns probability measure
def softmax(Z):
    ex = np.exp(Z)
    return ex/np.sum(ex, axis=0)

activations = {
    'sigmoid' :sigmoid,
    'tanh' : tanh,
    'relu' : relu,
    'softmax' : softmax
}

#################################
# Activation Function Derivatives
#################################

# returns derivative of sigmoid function
def sigmoid_prime(Z):
    return sigmoid(Z)*(1 - sigmoid(Z))

# returns derivative of tanh function
def tanh_prime(Z):
    return 1 - np.square(tanh(Z))

# returns derivative of relu function
def relu_prime(Z):
    Z[Z<=0] = 0
    Z[Z>0] = 1
    return Z

activation_derivatives = {
    'sigmoid' :sigmoid_prime,
    'tanh' : tanh_prime,
    'relu' : relu_prime
}

##############################
# Loss Functions
##############################
# reimpliment if results are not good
"""
Y_h denotes the output of the neural network
Y denotes the labels(correct) predictions
"""
# returns the cost of the neural network using L2 Loss
def squared_error(Y_h, Y):
    m = Y.shape[1]
    np.log(Y_h)
    square_comp = np.square(Y_h, Y)/2
    sum_layers = np.sum(square_comp, axis=0)
    mean = np.mean(sum_layers)
    return np.squeeze(mean)

# returns the cost of the neural network using Log Loss
def log_loss(Y_h, Y):
    m = Y.shape[1]
    log_comp = np.multiply(Y, np.log(Y_h)) + np.multiply((1-Y), np.log((1-Y_h)))
    sum_layers = - np.sum(log_comp, axis=0)
    mean = np.mean(sum_layers)
    return np.squeeze(mean)

# returns the cost of the neural network using Cross Entropy
def cross_entropy(Y_h, Y):
    m = Y.shape[1]
    log_comp = np.multiply(Y, np.log(Y_h))
    sum_layers = - np.sum(log_comp, axis=0)
    mean = np.mean(sum_layers)
    return np.squeeze(mean)

losses = {
    'squared_error': squared_error,
    'cross_entropy': cross_entropy,
    'log_loss': log_loss
}

##############################
# Output Derivatives Functions
##############################

def output_error(Y_h, Y, loss_function='log_loss'):
    loss_string = error_command[loss_function]
    command_string = loss_string %('Y_h', 'Y')
    # print(command_string)
    result = eval(command_string)
    return result

error_command = {
    'log_loss': '%s - %s',
    'cross_entropy': '(1 - %s) * %s'
}

##############################
# Training Functions
##############################

# returns a dictionary of parameters (weight, biases) accoridng to model
def define_parameters(model):
    paramaters = dict()
    layers = len(model)-1
    n_temp = n_x # number of neurons in the previous layer
    for layer in range (layers):
        pos = str(layer+1)
        # define dictionary keys for each weight/bias matrix accoridng to layer
        key_w, key_b = ('W' + pos ), ('b' + pos)
        n_h = model[layer+1] # depth of the layer
        weights = np.random.randn(n_h, n_temp)
        biases = np.zeros((n_h, 1))
        paramaters[key_w] = weights
        paramaters[key_b] = biases
        n_temp = n_h
    return paramaters

# returns a one hot encoded np matrix for classification
def one_hot_encoder(Y, n):
    m = Y.shape[1]
    result = np.zeros((n,m))
    for i in range(m):
        index = Y[0,i]
        result[index,i] = 1
    return result

# performs forward propagtion
def forward_propagate(X, parameters, layers, a_function='relu', o_function='sigmoid'):
    activate_h = activations[a_function] # activation function for hidden layers
    activate_o = activations[o_function] # activation function for output layer
    A = X # current layer matrix(activation)
    m = X.shape[1]
    neuron_cache = dict()
    neuron_cache['A0'] = X
    neuron_cache['activation'] = a_function
    neuron_cache['output'] = o_function
    for layer in range (layers):
        pos = str(layer+1)
        # define dictionary keys for each neuron/activation matrix accoridng to layer
        key_z, key_a = ('Z' + pos), ('A' + pos)
        key_w, key_b = ('W' + pos),('b' + pos) # used to access paramaters for network
        W, b = parameters[key_w], parameters[key_b]
        Z = np.dot(W, A) + b
        if not layer == (layers - 1):  # conditional on if hidden/output layers is being calculated
            A = activate_h(Z)
        else:
            A = activate_o(Z)
        neuron_cache[key_z], neuron_cache[key_a] = Z, A # save outputs into nueron_cache
    neuron_cache.update(parameters)
    assert(A.shape[1] == m)
    assert(A.shape[0] == n_y)
    return A, neuron_cache # final output and neuron_cache returned

# compute the cost of the neural network using selected loss function
def calculate_cost(Y, data, layers, loss_function='cross_entropy'):
    Y_h, cache = data
    cost_function = losses[loss_function]
    return cost_function(Y_h, Y)

# performs backwards propagtion
def backwards_propagate(X, Y, cache, layers,  l_function='log_loss'):
    Y_h, cache = cache
    a_derivative = activation_derivatives[cache['activation']]
    gradient_updates = dict()
    for layer in range(layers, 0, -1):
        pos = layer
        if (layer == layers):
            dZ =  output_error(Y_h, Y, l_function)
            E = dZ # donetes error
            gradient = backwards_propagate_helper(X, Y, cache, pos, dZ)
        else:
            key_W, key_Z = 'W' + str(pos+1) , 'Z' + str(pos)
            W, Z = cache[key_W], cache[key_Z]
            g_prime_Z = a_derivative(Z)
            dZ = np.dot(W.T, E) * g_prime_Z
            E = dZ
            gradient = backwards_propagate_weights(X, Y, cache, pos, dZ)
        gradient_updates.update(gradient)
    return gradient_updates

# performs weights and biases gradient calculations
def backwards_propagate_weights(X, Y, cache, pos, dZ):
    update = []
    m = X.shape[1]
    key_db, key_dW = 'db' + str(pos), 'dW' + str(pos)
    key_A = 'A' + str(pos-1)
    A = cache[key_A]
    dW = np.dot(dZ, A.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    result = dict([(key_dW, dW), (key_db, db)])
    return result

# updates neural network paramaters accoridng to gradients and learning rate
def update_params(params, gradients, layers, learning_rate=0.0005):
    alpha = learning_rate
    params_keys = set(params.keys())            # get keys from paramater dictionary
    for (key, dP) in gradients.items():
        key = key[1:]                    # remove  letter 'd' to retrieve weights/biases
        if (key in params_keys):
            params[key] = params[key] - alpha * dP
    return params
