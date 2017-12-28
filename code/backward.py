import numpy as np
import relu

def backprop(Z, dA, prevA, W, m):
    # dZ = dA * relu.derivative(Z)
    dZ = relu.backward(dA, Z)
    prev_dA = np.dot(np.transpose(W), dZ)
    dW = 1. / m * np.dot(dZ, np.transpose(prevA))
    db = 1. / m * np.sum(dZ, axis = 1, keepdims = True)
    return dW, db, prev_dA

def update_parameters(learning_rate, dW, db, W, b):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b
