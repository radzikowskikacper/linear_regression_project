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

W[1] = np.array([[-1.25878499, -0.83972071],
 [-1.08487785, -0.58338218]])

W[2] = np.array([[ 8.86772127, -9.51831613]])

b[1] = np.array([[-8.81354381],
 [ 7.16434734]])

b[2] = np.array([[ 3.39072051]])

activation_fun = {'forward': sigmoid, 'backward': sigmoid_backward}

sanity_check(L, W, b, activation_fun, function_to_estimate.function1)
print_metrics(layers, W, b, activation_fun)
