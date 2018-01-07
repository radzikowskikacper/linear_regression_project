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

W[1] = np.array([[ 1.03724227,  0.21805712],
 [ 0.99564428, -0.18942564]])

W[2] = np.array([[ 0.90711433, -0.89808267]])

b[1] = np.array([[-1.59213156],
 [-1.320275  ]])

b[2] = np.array([[-0.20060263]])

activation_fun = {'forward': relu, 'backward': relu_backward}

sanity_check(L, W, b, activation_fun, function_to_estimate.function1)
print_metrics(layers, W, b, activation_fun)
