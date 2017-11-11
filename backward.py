import numpy as np

def calculate_gradient(dZ, prevA, W, b, m):
    dW = 1. / m * np.dot(dZ * np.transpose(prevA))
    db = 1. / m * np.sum(dZ, axis = 1, keepdims = True)
    dPrevA = np.dot(np.transpose(W), dZ)
    return dW, db, dPrevA
