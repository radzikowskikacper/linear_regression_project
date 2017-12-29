import numpy as np
import relu

def backprop_on_layer(Z, dA, prevA, W, m):
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

def backprop(L, m, learning_rate, A, dA, W, b, Z):
    #Lth (last) layer:
    prevA = A[L - 1]
    dZ = dA
    prev_dA = np.transpose(W[L]) * dZ
    dW = (1. / m) * np.dot(dZ, np.transpose(prevA))
    db = 1. / m * np.sum(dZ, axis = 1, keepdims = True)
    W[L], b[L] = update_parameters(learning_rate, dW, db, W[L], b[L])

    #from (L-1)th to 1st layer:
    for l in reversed(range(1, L)):
        prevA = A[l - 1]
        dW, db, prev_dA = backprop_on_layer(Z[l], prev_dA, prevA, W[l], m)
        W[l], b[l] = update_parameters(learning_rate, dW, db, W[l], b[l])
    return W, b