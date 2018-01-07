import numpy as np

import function_to_estimate
from evaluate import print_metrics, sanity_check
from sigmoid import sigmoid, sigmoid_backward
from leaky_relu import leaky_relu, leaky_relu_backward
from relu import relu, relu_backward

layers = [2, 20, 20, 1]
L = len(layers) - 1  # number of layers - input layer doesn't count
