import numpy as np

def forward(X, L, W, b, activate_pointer):
    A = {0: X}
    Z = dict()

    for l in range(1, L):
        Z[l] = np.dot(W[l], A[l - 1]) + b[l]
        A[l] = activate_pointer(Z[l])

    Z[L] = np.dot(W[L], A[L - 1]) + b[L]
    A[L] = Z[L]

    return A, Z

def calculate_cost(out, trY):
    return np.sum(np.power(out - trY, 2))
