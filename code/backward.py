import numpy as np
import relu

def backprop(Z, dA, prevA, W, m):

    dZ = np.transpose(W) * dA * relu.derivative(Z)

    dW = 1. / m * np.dot(dZ, np.transpose(prevA))
    db = 1. / m * np.sum(dZ, axis = 1, keepdims = True)
    dPrevA = np.dot(np.transpose(W), dZ)
    return dW, db, dPrevA

def update_parameters(learning_rate, dW, db, dA):
    pass