import numpy as np

def backprop_on_layer(Z, dA, prevA, W, m, backward_activation_function):
    dZ = backward_activation_function(dA, Z)
    prev_dA = np.dot(np.transpose(W), dZ)
    dW = 1. / m * np.dot(dZ, np.transpose(prevA))
    db = 1. / m * np.sum(dZ, axis = 1, keepdims = True)
    return dW, db, prev_dA

def update_parameters(learning_rate, dW, db, W, b):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b

def backprop_with_update(L, m, learning_rate, A, dA, W, b, Z, backward_activation_function):
    #Lth (last) layer:
    prevA = A[L - 1]
    dW, db, prev_dA = backprop_on_layer(Z[L], dA, prevA, W[L], m, lambda dA, Z: dA)
    W[L], b[L] = update_parameters(learning_rate, dW, db, W[L], b[L])

    #from (L-1)th to 1st layer:
    for l in reversed(range(1, L)):
        prevA = A[l - 1]
        dW, db, prev_dA = backprop_on_layer(Z[l], prev_dA, prevA, W[l], m, backward_activation_function)
        W[l], b[l] = update_parameters(learning_rate, dW, db, W[l], b[l])
    return W, b