# OpenML

OpenML is a Machine Learning Framework designed to be intuitive and understandable for those new to machine learning. The project focuses on the use of neural networks for supervised learning. 

# Installation

Modules:
* numpy
* sklearn (optional)
* matplotlib (optional)

The modules can be installed by running the following command in a CLI environment: (Assumes PATH defined for python)

``` 
pip install <module_name> 
```
# Setup

In order to set a basic neural network, we will use the neural_network.py class to create an instatnce of the NN Class.
This can be done by creating an new python file and importing NN. First we need to set up our training data and lables to feed into the neural network. For the purpose of this example, we will be using the Iris dataset from the sklearn library.

```python
# import supporting libraries
import numpy as np
from sklearn.datasets import load_iris
from neural_network import NN 

dataset = load_iris()
training_data = dataset.data.T
m = training_data.shape[1]
label_data = dataset.target.T
```
One thing to notice here is the use of '.T' on both the training and label data. This operation is a transpose on the ndarray done to place the individual training examples in unique columns. This step requires knowledge of how the data is initially set up. Next we will set up the neural network model.

```python
# Setup model for NN
n_x = training.shape[0]   # number of input features of NN
n_y = 3                   # number of clasess of NN
model = [n_x, 3, 6, n_y]  # NN layers with number of neurons 

# instanstiate NN Object
classifier = NN(model, X, Y, one_hot=True)

```
The neural network is setup using a list with each element as the desired number of neurons in the given layer. The length of the model is how 'deep' the neural network is. Worth noting is the inclusion of the 'one_hot' argument in the neural network class. This is typically used when the NN is classifying more than 2 classes (in this case 3). More information on this can be found [here](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f). 
Next we will define our NN functions.

```python
# set up NN training functions
classifier.activation_function('relu')
classifier.output_function('softmax')
classifier.loss_function('log_loss')
```
The initiaztion methods here accept string arguments to select the function to be used in the neural network. If no functions are initialized, the NN will default to the same configuration as seen above. Lastly, we will train the neural network and tests its accuracy on its inputs.

```python
# train neural network and compute accuracy
classifier.train()
classifier.predict()
```

We now have a fully functioning neural network! 





