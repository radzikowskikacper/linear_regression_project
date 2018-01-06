import numpy as np

# Example function to be estimated by NN:
#  y = sin(x1) * cos(x2)
def function1_old(X):
    return np.multiply(2 * np.sin(X[0, :]) + 5, 0.8 * np.cos(X[1, :]) - 3)

def function1(X):
    return 10 * np.sin(X[0, :] / 5) * np.tanh(X[1, :])

# Example function to be estimated by NN:
#  y = x1^2 + x2^2
def function2(X):
    return np.reshape(np.multiply(X[0, :], X[0, :]) + np.multiply(X[1, :], X[1, :]), (1, -1))
