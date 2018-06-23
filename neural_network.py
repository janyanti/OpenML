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
    def __init__(self, model, X, Y, learning_rate=0.005, one_hot=False):
        self.model = model
        self.X = X
        self.Y = Y
        if one_hot : self.Y = one_hot_encoder(Y, model[-1])
        self.layers = len(self.model) - 1
        self.params = define_parameters(self.model)
        self.activation_func = None
        self.output_func = None
        self.loss_func = None

    def activation_function(self, activation='relu'):
        self.activation_func = activations[activation]
        self.a_derivative = activation_derivatives[activation]

    def output_function(self, output='sigmoid'):
        self.output_func = activations[output]
        self.a_derivative = activation_derivatives[output]

    def loss_function(self, loss='log_loss'):
        self.loss_func = losses[loss]

    def train(self, iterations=100000, cost_interval=100):
        for i in range (iterations):
            self.feed = forward_propagate(self.X, self.params, self.layers, self.activation_func, self.output_func)
            self.cost = calculate_cost(self.Y, self.feed, self.layers, self.loss_func)
            self.grads = backwards_propagate(self.X, self.Y, self.feed, self.layers, self.a_derivative)
            self.params = update_params(self.params, self.grads,self.layers)
            if (i % cost_interval == 0) :
                print(self.cost)

        self.final_params = (self.params, self.activation_func, self.output_func)
        print(self.cost)
        print(self.params)

    def predict(self):
        num_outputs = self.model[-1]
        self.results = forward_propagate(self.X, self.params, self.layers, self.activation_func, self.output_func)
        Y_h = self.results[0]
        print(np.round(np.subtract(Y_h, self.Y)))
        if self.output_func == 'sigmoid' or num_outputs == 1:
            predict_logistic_regression(Y_h, self.Y)
        elif self.output_func == 'softmax' or num_outputs > 1:
            predict_multi_class_classifier(Y_h, self.Y)

#################################
# Activation Functions
#################################

# activation function that returns a value between 0 and 1
def sigmoid_func(Z):
    return (1/(1+np.exp(-Z)))

# activation function that returns a value between -1 and 1
def tanh_func(Z):
    return np.tanh(Z)

# activation function that returns the identity of input or 0 if less than 0
def relu_func(Z):
    return np.maximum(0,Z)

# activation function for multiclass outputs | returns probability measure
def softmax_func(Z):
    ex = np.exp(Z)
    return ex/np.sum(ex, axis=0)

activations = {
    'sigmoid' :sigmoid_func,
    'tanh' : tanh_func,
    'relu' : relu_func,
    'softmax' : softmax_func
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
# returns derivative of softmax_func | NEEDS OVERHAULS
def softmax_prime(Z):
    return Z

activation_derivatives = {
    'sigmoid' :sigmoid_prime,
    'tanh' : tanh_prime,
    'relu' : relu_prime,
    'softmax': softmax_prime
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
    n_temp = model[0] # number of neurons in the previous layer
    for layer in range (layers):
        pos = str(layer+1)
        # define dictionary keys for each weight/bias matrix accoridng to layer
        key_w, key_b = ('W' + pos ), ('b' + pos)
        n_h = model[layer+1] # depth of the layer
        weights = np.random.randn(n_h, n_temp)
        biases = np.random.randn(n_h, 1)
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
def forward_propagate(X, parameters, layers, a_function=relu_func, o_function=sigmoid_func):
    activate_h = a_function # activation function for hidden layers
    activate_o = o_function # activation function for output layer
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
        if layer == (layers - 1):  # conditional on if hidden/output layers is being calculated
            A = activate_o(Z)
        else:
            A = activate_h(Z)
        neuron_cache[key_z], neuron_cache[key_a] = Z, A # save outputs into nueron_cache
    neuron_cache.update(parameters)
    assert(A.shape[1] == m)
    return A, neuron_cache # final output and neuron_cache returned

# compute the cost of the neural network using selected loss function
def calculate_cost(Y, data, layers, loss_function=cross_entropy):
    Y_h, cache = data
    return loss_function(Y_h, Y)

# performs backwards propagtion
def backwards_propagate(X, Y, cache, layers, derivative, l_function='log_loss'):
    Y_h, cache = cache
    a_derivative = derivative
    gradient_updates = dict()
    for layer in range(layers, 0, -1):
        pos = layer
        if (layer == layers):
            dZ =  output_error(Y_h, Y, l_function)
            E = dZ # donetes error
            gradient = backwards_propagate_params(X, Y, cache, pos, dZ)
        else:
            key_W, key_Z = 'W' + str(pos+1) , 'Z' + str(pos)
            W, Z = cache[key_W], cache[key_Z]
            g_prime_Z = a_derivative(Z)
            dZ = np.dot(W.T, E) * g_prime_Z
            E = dZ
            gradient = backwards_propagate_params(X, Y, cache, pos, dZ)
        gradient_updates.update(gradient)
    return gradient_updates

# performs weights and biases gradient calculations
def backwards_propagate_params(X, Y, cache, pos, dZ):
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

##############################
# Prediction Functions
##############################

# computes preditction query from trained neural network/evaluates accuracy
def predict_logistic_regression(Y_h, Y):
    assert(Y_h.shape == Y.shape)
    m = max(Y.shape[0], Y.shape[1])
    Y_h[Y_h<0.5] = 0
    Y_h[Y_h>=0.5] = 1
    result = compute_prediction_accuracy(Y_h, Y)
    print(result)

def predict_multi_class_classifier(Y_h, Y):
    m = max(Y.shape[0], Y.shape[1])
    encoded_Y_h = softmax_converter(Y_h)
    print(Y.shape, encoded_Y_h.shape)
    assert(encoded_Y_h.shape == Y.shape)
    result = compute_prediction_accuracy(encoded_Y_h, Y)
    print(result)

# converts a softmax matrix to a matrix of only zeros and ones
def softmax_converter(Y_h):
    maxes = np.amax(Y_h, axis=0)
    bool_matrix = np.apply_along_axis(np.equal, 1, Y_h, maxes)
    one_hot_encoded = bool_matrix.astype(int)
    return one_hot_encoded

def compute_prediction_accuracy(Y_h, Y):
    m = Y.shape[1]
    num_incorrect = np.count_nonzero(Y_h - Y)/2
    num_correct = m - num_incorrect
    print(num_correct, m)
    accuracy = (num_correct/m) * 100
    accuracy_string = '%.3f %s of predictions correct' %(accuracy, '%')
    return accuracy_string
