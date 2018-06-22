##################################################
# Joel Anyanti | CMU
# Python classifier
##################################################
# imports
from matplotlib import pyplot as plt
import numpy as np
import os
import csv
import math

##################################################
# Helper Functions
##################################################

# format number strings into integer values
def format_numbers(list, type = 'int'):
    result = []
    for (i, elem) in enumerate(list):
        try:
            if type == 'int':
                list[i] = int(elem)
            else:
                list[i] = float(elem)
        except Exception as e:
            continue

# parse data into nested list format with number/type arguments
def parse_data(filename, header = False):
    result = []
    with open(filename) as data:
        reader = csv.reader(data)
        if header: next(reader, None)              #Skips header
        for row in reader:
            format_numbers(row, 'float')
            result.append(row)
    result = np.array(result)
    result = np.rot90(result, axes=(0,1))
    result = np.flip(result, 0)
    print(result)
    return result

# Loss functions for calculating cost
def squared_error(pred, target):
    return ((pred - target)**2)/2

# Takes in t
def cross_entropy(X, y):
    pass

# generates target vectors
def target_vector(target, num_labels):
    result = np.zeros(num_labels)
    result[target] = 1
    return result

# activation function that returns a value between 0 and 1
def sigmoid(z):
    return (1/(1+np.exp(-z)))
# activation function that returns a value between -1 and 1
def tanh(z):
    return np.tanh(z)

# activation function that returns the identity of input or 0 if less than 0
def relu(z):
    return max(0,z)

# activation function for multiclass outputs | returns probability measure
def softmax(args):
    ex= np.exp(args)
    sum_ex = np.sum( np.exp(args))
    return ex/sum_ex

# used for derivatives of softmax function
def kronecker_delta(i, j):
    if i == j: return 1
    return 0

def predict(a1, w1, b):
    z = a1*w1 + b
    result = sigmoid(z)
    return result

# used for random generation of weights and biases (between -1 and 1)
def generate_weights(hidden_layers, layer_width, num_inputs):
    result = []
    for i in range (hidden_layers):
        weight = np.random.randn(layer_width, num_inputs)
        result.append (weight)
    return np.array(result)

def generate_bias(hidden_layers, layer_width):
    for i in range (hidden_layers):
        biases = np.random.randn(hidden_layers,layer_width)
        return biases

def forward_propagate (layer_width, features, weight_matrix, bias_matrix):
    result = []
    for i in range(layer_width):
        weight = weight_matrix[0][i]
        bias = bias_matrix[0][i]
        dot_product = np.dot(weight, features)
        z = np.add(dot_product, bias)
        temp_neuron = sigmoid(z)
        result.append(temp_neuron)
        # print(dot_product)
    #print(result)
    return np.array(result)


##################################################
# Main
##################################################

# initialize data from csv file
filename = 'iris.csv'
test_data = parse_data(filename, True)
hidden_layers = 1
num_labels = 3
layer_width = 3
# w1 = np.random.randn()
# b = np.random.randn()
#print(test_data)


def train_model(data):
    num_inputs = len(data[0])-1
    targets = []
    weight_matrix = generate_weights(hidden_layers, layer_width, num_inputs)
    print(weight_matrix, 'w')
    bias_matrix = generate_bias(hidden_layers, layer_width)
    hidden_neurons = []
    for (i, elem) in enumerate(data):
        targets.append(elem[-1])
        features = np.delete(elem, -1)
        # print(test_number)
        hidden_neurons.append(forward_propagate(layer_width, features, weight_matrix, bias_matrix))
    hidden_neurons = np.array(hidden_neurons)
    # print(hidden_neurons)
    result_softmax, predictions = [], []
    for i in range (len(hidden_neurons)):
        result_softmax.append(softmax(hidden_neurons[i]))
        predictions.append(np.argmax(hidden_neurons[i]))

    result_softmax = np.array(result_softmax)
    total_loss = 0
    for (i, elem) in enumerate (targets):
        prediction, target = predictions[i], targets[i]
        loss = squared_error(prediction, target)
        total_loss += loss
    print(total_loss)
        # prediction = predict(test_number[0], w1, b)
    #     loss = cost(prediction, target)
    #     total_loss += loss
    # print(total_loss)

if __name__ == '__main__':
    train_model(test_data)
