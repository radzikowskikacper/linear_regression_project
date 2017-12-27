import numpy as np

def relu(Z):
    return np.maximum(0, Z)

def backward(dA, Z):
    return np.copy(dA) * derivative(Z)

def derivative(Z):
    dZ = np.ones(np.shape(Z))
    dZ[Z<=0]=0
    return dZ
