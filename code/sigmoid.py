import numpy as np

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def sigmoid_backward(dA, Z):
    return dA * derivative(Z)

def derivative(Z):
    S = sigmoid(Z)
    return S * (1-S)
