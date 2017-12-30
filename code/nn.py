import numpy as np
from scipy.stats import pearsonr, spearmanr

import function_to_estimate
from backward import backprop_with_update
from forward import forward, calculate_cost
from leaky_relu import leaky_relu, leaky_relu_backward
from relu import relu, relu_backward
from sigmoid import sigmoid, sigmoid_backward
from data_gen import generate_data_sets

layers = [2, 10, 10, 1] # number of units in each layer (layers[0] - input layer)
L = len(layers) - 1  # number of layers - input layer doesn't count
number_of_iterations = 200000
learning_rate = 0.05

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
    W[l] = np.random.randn(layers[l], layers[l-1]) * 0.01 # or / np.sqrt(layers[l])
    b[l] = np.zeros((layers[l], 1))

trX, trY, tsX, tsY, validation_data = generate_data_sets(train_data_min, train_data_max, train_data_step, test_data_min, test_data_max, test_data_step)

m = np.shape(trX)[1] # number of training examples

#choose activation function
leaky_relu_fun = {'forward': leaky_relu, 'backward': leaky_relu_backward}
relu_fun = {'forward': relu, 'backward': relu_backward}
sigmoid_fun = {'forward': sigmoid, 'backward': sigmoid_backward}
activation_fun = sigmoid_fun

#training
for i in range(1, number_of_iterations+1):
    #forward propagation through all the layers
    trA, trZ = forward(trX, L, W, b, activation_fun['forward'])
    trcost = calculate_cost(trA[L], trY)
    tsA, tsZ = forward(tsX, L, W, b, activation_fun['forward'])
    tscost = calculate_cost(tsA[L], tsY)

    if i%50 == 0:
        print('Iteration: ', i, '   Training cost: {:.5f} Testing cost: {:.5f}'.format(trcost, tscost))

    #back propagation through all the layers
    dA = trA[L] - trY #initialization (cost derivative)
    W, b = backprop_with_update(L, m, learning_rate, trA, dA, W, b, trZ, activation_fun['backward'])

def predict(X):
    A, _ = forward(X, L, W, b, activation_fun['forward'])
    Y = function_to_estimate.function1(X)
    print(X)
    print("real value: ", Y)
    print("prediction: ", A[L])

print("MODEL: \n\n")
print("W: ", W)
print("b: ", b)

# sanity check
print("PREDICTIONS: \n\n")
predict(np.array([[1], [2]]))
predict(np.array([[5], [5]]))
predict(np.array([[2], [8]]))
predict(np.array([[2], [2]]))
predict(np.array([[-1], [2]]))
predict(np.array([[-5], [-5]]))
predict(np.array([[-3], [-9]]))
predict(np.array([[1], [-1]]))
predict(np.array([[1.5], [-1.8]]))
predict(np.array([[-2.2], [2.4]]))
predict(np.array([[-2.1], [-3.5]]))
predict(np.array([[0.3], [0.5]]))

tsA, _ = forward(tsX, L, W, b, activation_fun['forward'])

from evaluate import evaluate_model
evaluate_model(layers, W, b, activation_fun)
