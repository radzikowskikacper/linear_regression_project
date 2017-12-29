import numpy as np

def leaky_relu(Z):
    return np.maximum(0.1*Z, Z)

def leaky_relu_backward(dA, Z):
    return np.copy(dA) * derivative(Z)

def derivative(Z):
    dZ = np.ones(np.shape(Z))
    dZ[Z <= 0] = 0.1
    return dZ
