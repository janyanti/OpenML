#################################
# Joel Anyanti | 06/20/2018
# Carnegie Mellon University
#################################
#################################
# imports
#################################
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from drawnow import *
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X_train = np.vstack([img.reshape(-1,) for img in mnist.train.images])
y_train = mnist.train.labels
print(y_train.shape)

X_train = X_train.T
y_train = y_train.T
print(X_train.shape, y_train.shape)

# with open('data.txt', 'w') as file:
#     file.write('X:' + str(X_train))
#     file.write('y' + str(y_train))
#     file.close()


"""
'model' defines the characteristics of the nueral network
the first number in the list is the number of input neurons
or features in the nueral network. The last number in the list is
the number of desired output neurons. The numbers in the middle
represent the neural networks hideen layers. This model will allow
us to define the paramaters of the neural network"""

#################################
# Model Variables
#################################
# data = load_iris()
data = load_breast_cancer()
# print(data2.data.shape, data2.target.shape)
X = data.data
X = X.T

# test = [[1,0,0],[0,1,0],[0,0,1]]
# test = np.array(test, dtype='float')

# Number of training examples
m = X_train.shape[1]
# Y = np.reshape(data.target, (1,m))
# number of input/output neurons
n_x = X_train.shape[0]
# n_y = Y.shape[0]
n_y = 10

# nueral network model
model = [n_x, 6, 8, 5, n_y]


# number of layers
layers = len(model) - 1


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
    ex = np.exp(Z - np.max(Z))
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
    result = eval(command_string)
    return result

error_command = {
    'log_loss': '%s - %s',
    'cross_entropy': '(1 - %s) * %s'
}

# returns a dictionary of parameters (weight, biases) accoridng to model
def define_parameters(model):
    paramaters = dict()
    layers = len(model)-1
    n_temp = n_x # number of neurons in the previous layer
    for i in range (layers):
        # define dictionary keys for each weight/bias matrix accoridng to layer
        key_w, key_b = ('W' + str(i+1)),('b' + str(i+1))
        n_h = model[i+1] # depth of the layer
        weights = np.random.randn(n_h, n_temp)
        biases = np.random.randn(n_h, 1)
        paramaters[key_w] = weights
        paramaters[key_b] = biases
        n_temp = n_h
    return paramaters


def one_hot_encoder(Y, n):
    m = Y.shape[1]
    result = np.zeros((n,m))
    for i in range(m):
        val = Y[0,i]
        result[val,i] = 1
    return result

# performs forward propagtion
def forward_propagate(X, parameters, layers, a_function='tanh', o_function='softmax'):
    activate_h = activations[a_function] # activation function for hidden layers
    activate_o = activations[o_function] # activation function for output layer
    A = X # current layer matrix(activation)
    m = X.shape[1]
    neuron_cache = dict()
    neuron_cache['A0'] = X
    neuron_cache['activation'] = a_function
    neuron_cache['output'] = o_function
    for i in range (layers):
        pos = str(i+1)
        # define dictionary keys for each neuron/activation matrix accoridng to layer
        key_z, key_a = ('Z' + pos), ('A' + pos)
        key_w, key_b = ('W' + pos),('b' + pos) # used to access paramaters for network
        W, b = parameters[key_w], parameters[key_b]
        Z = np.dot(W, A) + b
        if not i == (layers - 1):  # conditional on if hidden/output layers is being calculated
            A = activate_h(Z)
        else:
            A = activate_o(Z)
        neuron_cache[key_z], neuron_cache[key_a] = Z, A # save outputs into nueron_cache
    neuron_cache.update(parameters)
    assert(A.shape[1] == m)
    assert(A.shape[0] == n_y)
    return A, neuron_cache # final output and neuron_cache returned

def calculate_cost(Y, data, layers, loss_function='cross_entropy'):
    Y_h, cache = data
    cost_function = losses[loss_function]
    return cost_function(Y_h, Y)


def backwards_propagate(X, Y, cache, layers,  l_function='log_loss'):
    Y_h, cache = cache
    a_derivative = activation_derivatives[cache['activation']]
    #o_derivative = activation_derivatives[cache['output']]
    gradient_updates = dict()
    for i in range(layers, 0, -1):
        pos = i
        if (i == layers):
            dZ =  output_error(Y_h, Y, l_function)
            E = dZ # donetes error
            gradient = backwards_propagate_helper(X, Y, cache, pos, dZ)
        else:
            key_W, key_Z = 'W' + str(pos+1) , 'Z' + str(pos)
            W, Z = cache[key_W], cache[key_Z]
            g_prime_Z = a_derivative(Z)
            dZ = np.dot(W.T, E) * g_prime_Z
            E = dZ
            gradient = backwards_propagate_helper(X, Y, cache, pos, dZ)
        gradient_updates.update(gradient)
    return gradient_updates

# performs forward propagtion
def backwards_propagate_helper(X, Y, cache, pos, dZ):
    update = []
    m = X.shape[1]
    key_db, key_dW = 'db' + str(pos), 'dW' + str(pos)
    key_A = 'A' + str(pos-1)
    A = cache[key_A]
    dW = np.dot(dZ, A.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    result = dict([(key_dW, dW), (key_db, db)])
    return result

def update_params(params, gradients, layers, learning_rate=0.0005):
    alpha = learning_rate
    params_keys = set(params.keys())
    for (key, value) in gradients.items():
        key_params = key[1:]
        if (key_params in params_keys):
            params[key_params] = params[key_params] - alpha * value
    return params


# X = np.random.randn(n_x, m)
# Y = np.random.randn(model[-1],m)

params = define_parameters(model)
# y_train = one_hot_encoder(y_train, n_y)
cost_list = []
             #Set y min and max values based on known voltage quantity
plt.grid(True)
plt.xlabel("iterations")
# plt.scatter(cost_list, '-', label='Cost'
plt.axis([0, 50000, 0, 10])
plt.ion()


print(X_train)
for i in range (10000):
    cost_list = []
    index = []
    data = forward_propagate(X_train, params, layers)
    cost = calculate_cost(y_train, data, layers)
    grads = backwards_propagate(X_train, y_train, data, layers)
    params = update_params(params, grads, layers)
    if (i % 1000 == 0) :
        print(cost)
        plt.scatter(i, 2*cost)
        plt.pause(0.002)
    plt.show()
print(cost)
print(params)

#result = output_error(data[0], Y)
#print(result)
#print(cost)


"""
print('Relu:', relu_prime(L1))
print('Sigmoid', sigmoid_prime(L1))
print('Tanh', tanh_prime(L1))"""
#print(data)
