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

# layers = [2, 2, 1]  # number of units in each layer (layers[0] - input layer)
layers = [2, 2, 3, 3, 5, 5, 3, 3, 2, 2, 1]  # number of units in each layer (layers[0] - input layer)
# layers = [2, 20, 20, 1]  # number of units in each layer (layers[0] - input layer)
# layers = [2, 100, 100, 1]  # number of units in each layer (layers[0] - input layer)
L = len(layers) - 1  # number of layers - input layer doesn't count
number_of_iterations = 10000
learning_rate = 0.05
save_cost_plot_after_epoch = list([50, 100, 1000])

train_data_min = -20
train_data_max = 20
train_data_step = 1
test_data_min = -20
test_data_max = 20
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

fun_to_estimate = function_to_estimate.function1
plot_function(data[0,:], data[1,:], fun_to_estimate)

m = np.shape(trX)[1]  # number of training examples

# choose activation function
leaky_relu_fun = {'forward': leaky_relu, 'backward': leaky_relu_backward}
relu_fun = {'forward': relu, 'backward': relu_backward}
sigmoid_fun = {'forward': sigmoid, 'backward': sigmoid_backward}
activation_fun = sigmoid_fun

trcosts, tscosts = list(), list()
tr_hist = list()

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

        if i % 50 == 0:
            tr_hist_line = 'Iteration: {}   Training cost: {:.5f} Testing cost: {:.5f}'.format(i, trcost, tscost)
            tr_hist.append(tr_hist_line)
            print(tr_hist_line)
            plot_errors(trcosts, tscosts, 'Training cost', 'Validation cost', 'Cost chart', 'costs.png')
            if i in save_cost_plot_after_epoch:
                plot_errors(trcosts, tscosts, 'Training cost', 'Validation cost', 'Cost chart', 'costs_after_{}_epoch.png'.format(i))

        # back propagation through all the layers
        dA = trA[L] - trY  # initialization (cost derivative)
        W, b = backprop_with_update(L, m, learning_rate, trA, dA, W, b, trZ, activation_fun['backward'])
    if (input("Would you like to continue? [y/n]: ").upper() == 'Y'):
        iterations_start = iterations_stop
        number_of_iterations = int(input("How many addition iterations?"))
    else:
        continue_train = False

sanity_check(L, W, b, activation_fun, fun_to_estimate)
print_metrics(layers, W, b, activation_fun)
save_model(layers, W, b, activation_fun, tr_hist, number_of_iterations, learning_rate, fun_to_estimate, save_cost_plot_after_epoch)
