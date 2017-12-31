import numpy as np

import function_to_estimate
from backward import backprop_with_update
from data_gen import generate_data_sets, generate_data_helper
from evaluate import print_metrics, sanity_check
from forward import forward, calculate_cost
from leaky_relu import leaky_relu, leaky_relu_backward
from relu import relu, relu_backward
from saver import save_model
from sigmoid import sigmoid, sigmoid_backward
from plotter import plot_function, plot_errors

layers = [2, 100, 100, 1]  # number of units in each layer (layers[0] - input layer)
L = len(layers) - 1  # number of layers - input layer doesn't count
number_of_iterations = 200000
learning_rate = 0.05

train_data_min = -10
train_data_max = 10
train_data_step = 1
test_data_min = -10
test_data_max = 10
test_data_step = 1

# parameters initialization
np.random.seed(1)
W = {}
b = {}
for l in range(1, len(layers)):
    W[l] = np.random.randn(layers[l], layers[l - 1]) * 0.01  # or / np.sqrt(layers[l])
    b[l] = np.zeros((layers[l], 1))

data = generate_data_helper(train_data_min, train_data_max, train_data_step, test_data_min, test_data_max, test_data_step)
trX, trY, tsX, tsY, validation_data = generate_data_sets(data)

plot_function(data[0,:], data[1,:], function_to_estimate.function1)

m = np.shape(trX)[1]  # number of training examples

# choose activation function
leaky_relu_fun = {'forward': leaky_relu, 'backward': leaky_relu_backward}
relu_fun = {'forward': relu, 'backward': relu_backward}
sigmoid_fun = {'forward': sigmoid, 'backward': sigmoid_backward}
activation_fun = sigmoid_fun

trcosts, tscosts = list(), list()

# training
continue_train = True
iterations_start = 1
while continue_train:
    iterations_stop = iterations_start + number_of_iterations
    for i in range(iterations_start, iterations_stop):
        # forward propagation through all the layers
        trA, trZ = forward(trX, L, W, b, activation_fun['forward'])
        trcost = calculate_cost(trA[L], trY)
        tsA, tsZ = forward(tsX, L, W, b, activation_fun['forward'])
        tscost = calculate_cost(tsA[L], tsY)
        trcosts.append(trcost)
        tscosts.append(tscost)

        # if i % 50 == 0:
        print('Iteration: ', i, '   Training cost: {:.5f} Testing cost: {:.5f}'.format(trcost, tscost))
        plot_errors(trcosts, tscosts, 'Training cost', 'Validation cost', 'Cost chart', 'costs.png')

        # back propagation through all the layers
        dA = trA[L] - trY  # initialization (cost derivative)
        W, b = backprop_with_update(L, m, learning_rate, trA, dA, W, b, trZ, activation_fun['backward'])
    if (input("Would you like to continue? [y/n]: ").upper() == 'Y'):
        iterations_start = iterations_stop
        number_of_iterations = int(input("How many addition iterations?"))
    else:
        continue_train = False

sanity_check(L, W, b, activation_fun, function_to_estimate.function1)
print_metrics(layers, W, b, activation_fun)
save_model(layers, W, b, activation_fun)
