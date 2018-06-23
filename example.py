#################################
# Joel Anyanti | 06/20/2018
# Carnegie Mellon University
#################################

#################################
# imports
#################################
from neural_network import NN
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
#################################
# Main body
#################################

# defining training data and neural network model

# Load data from sklearn for Iris multiclass classification
data_1 = load_iris()
X_train_1 = data_1.data
X_train_1 = X_train_1.T
m = X_train_1.shape[1]
Y_train_1 = np.reshape(data_1.target.T, (1,m))
n_x = X_train_1.shape[0]
n_y = 3                                 # set value for number of output neurons
model_1 = [n_x, 4, 6, 4, n_y]

# Load data from sklearn for breast cancer logistic regression
data_2 = load_breast_cancer()
X_train_2 = data_2.data
X_train_2 = X_train_2.T
m = X_train_2.shape[1]
Y_train_2 = np.reshape(data_2.target.T, (1,m))
n_x = X_train_2.shape[0]
n_y = 1                                 # set value for number of output neurons
model_2 = [n_x, 4, 6, 4, n_y]

# Define neural networks

# Iris Mutli-Class Classifier
NN1 = NN(model_1, X_train_1, Y_train_1)
NN1.activation_function('relu')
NN1.output_function('softmax')
NN1.loss_function('cross_entropy')

# Breast Cancer Logistic Regression
NN2 = NN(model_2, X_train_2, Y_train_2)
NN2.activation_function('tanh')
NN2.output_function('sigmoid')
NN2.loss_function('log_loss')

# Train neural networks
NN1.train()
NN1.predict()
NN2.train()
NN2.predict()

#
