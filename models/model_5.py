import numpy as np

import function_to_estimate
from evaluate import print_metrics, sanity_check
from sigmoid import sigmoid, sigmoid_backward
from leaky_relu import leaky_relu, leaky_relu_backward
from relu import relu, relu_backward

layers = [2, 2, 1]
L = len(layers) - 1  # number of layers - input layer doesn't count

W={}
b={}

W[1] = np.array([[ 1.09796257,  0.19089885],
 [ 0.96220053, -0.15826102]])

W[2] = np.array([[ 0.97486294, -0.8774797 ]])

b[1] = np.array([[-1.54313208],
 [-1.17126357]])

b[2] = np.array([[-0.10768256]])

activation_fun = {'forward': leaky_relu, 'backward': leaky_relu_backward}

sanity_check(L, W, b, activation_fun, function_to_estimate.function1)
print_metrics(layers, W, b, activation_fun)
