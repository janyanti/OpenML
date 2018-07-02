# Todos

Neural network logic appears to be implemented properly .
Known errors appear to occur on istances where matrix values are arbitrarily close to 0.
Errors may be fixed by adding safeguards to fucntions when the values of the matrices approach this condition.


Suspected Functions to Fix (All functions in neural_network.py)
* cross_entropy
* log_loss
* relu_prime
* softmax_prime
* softmax 

# Procedure 

1. Log (print) resulting matrix values step by step to identify the root of the errors
2. Use the numpy library to test for various scenarios where matrix values are arbitrarily close to 0
3. Provide alternative values that allow for a functioniong neural network without compromising effectiveness
4. Implement refreshed functions that make use of modifications and test with example.py

