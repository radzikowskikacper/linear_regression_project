import numpy as np

#Example function to be estimated by NN:
#  y = sin(x1) * cos(x2)
def function1(X):
    return np.multiply(np.sin(X[0,]), np.cos(X[1,]))

#Example function to be estimated by NN:
#  y = x1^2 + x2^2
def function2(X):
    return np.multiply(X[0,], X[0,]) + np.multiply(X[1,], X[1,])
