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

In order to set a basic neural network, we will use the neural_network.py class to create an instatnce of the NN Class
This can be done by creating an new python file and importing NN.

```
# import nerual network class
from neural_network import NN

# Setup model for NN

n_x = 4                   # replace with the number of features/input neurons of the nerual network
n_y = 6                   # replace with the number of clasess/input neurons of the nerual network
model = [n_x, 3, 6, n_y]

# instanstiate NN Object
basic_classifier = NN(model, X, Y)

```




