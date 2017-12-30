import numpy as np
import itertools as ittools

#Example function to be estimated by NN:
#  y = sin(x1) * cos(x2)
def function1(X):
#     return function1(X[0, :], X[1, :])
    return np.multiply(2 * np.sin(X[0, :]) + 5, 0.8 * np.cos(X[1, :]) - 3)

# def function1(X, Y):
#     return np.multiply(2 * np.sin(X) + 5, 0.8 * np.cos(Y) - 3)

#Example function to be estimated by NN:
#  y = x1^2 + x2^2
def function2(X):
    return np.reshape(np.multiply(X[0, :], X[0, :]) + np.multiply(X[1, :], X[1, :]), (1, -1))

def function3(X):
    return X[0, :]# + X[1, :]

def generate_data(x1min, x1max, x1step, x2min, x2max, x2step):
    np.random.seed(1)
    arr = np.array(list(ittools.product(np.arange(x1min, x1max, x1step), np.arange(x2min, x2max, x2step))))
    arr = np.reshape(arr, (2, -1))
    arr = np.vstack((arr, function1(arr), function2(arr)))
    return arr
