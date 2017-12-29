import numpy as np

from random import shuffle
from forward import forward, calculate_cost
from backward import backprop
from function_to_estimate import generate_data
from relu import relu, relu_backward
from sigmoid import sigmoid, backward
import function_to_estimate

layers = [2, 4, 5, 1] # number of units in each layer (layers[0] - input layer)
L = len(layers) - 1  # number of layers - input layer doesn't count
W = {}
b = {}
number_of_iterations = 200000
learning_rate = 0.001
'''
num_of_train_examples = 100

np.random.seed(100) # to have the same results (testing) - remove in final solution
X = np.floor(np.random.rand(layers[0], num_of_train_examples) * 200 - 100) # inputs
# rows - features, in our example one example is two-dimentional vector
# columns - training example, above we have 1000 training examples
Y = function_to_estimate.function1(X) # outputs

'''
# parameters initialization
for l in range(1, len(layers)):
    W[l] = np.random.randn(layers[l], layers[l-1]) * 0.01 # or / np.sqrt(layers[l])
    b[l] = np.zeros((layers[l], 1))

data = generate_data(0, 1000, 1, 0, 1000, 1)
data_indexes = np.random.choice([0, 1, 2], p = [0.8, 0.1, 0.1], size = (1, data.shape[1]))
training_data = data[:, (data_indexes == 0)[0]]
testing_data = data[:, (data_indexes == 1)[0]]
validation_data = data[:, (data_indexes == 2)[0]]
trX = training_data[0:2,:]
trY = np.reshape(training_data[2,:], (1, -1))
tsX = testing_data[0:2,:]
tsY = np.reshape(testing_data[2,:], (1, -1))

m = np.shape(trX)[1] # number of training examples

#training
for i in range(0, number_of_iterations):
    #forward propagation through all the layers
    trA, trZ = forward(trX, L, W, b, sigmoid)
    trcost = calculate_cost(trA[L], trY)
    tsA, tsZ = forward(tsX, L, W, b, sigmoid)
    tscost = calculate_cost(tsA[L], tsY)

    print('Training cost: {:.2f} Testing cost: {:.2f}'.format(trcost, tscost))

    #back propagation through all the layers
    # dA = np.multiply(np.transpose(W[L - 1]), (A - Y)) # not sure - check if correct
    dA = trA[L] - trY #initialization (cost derivative)

    W, b = backprop(L, m, learning_rate, trA, dA, W, b, trZ, backward)

# print(W)
# print(b)

