import numpy as np

def forward(trX, L, W, b, activate_pointer):
    A = {0: trX.T}
    Z = dict()

    for l in range(1, L):
        Z[l] = np.dot(A[l - 1], W[l]) + b[l]
        A[l] = activate_pointer(Z[l])

    return A, Z

def calculate_cost():
    pass
