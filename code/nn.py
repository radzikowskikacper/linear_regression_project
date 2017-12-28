import numpy as np

from random import shuffle
from forward import forward, calculate_cost
from backward import backprop, update_parameters
from function_to_estimate import generate_data
from relu import relu
import function_to_estimate

layers = [2, 5, 4, 1] # number of units in each layer (layers[0] - input layer)
L = len(layers) - 1  # number of layers - input layer doesn't count
W = {}
b = {}
number_of_iterations = 200
learning_rate = 0.05

np.random.seed(100) # to have the same results (testing) - remove in final solution
X = np.floor(np.random.rand(2, 1000) * 100) # inputs
# rows - features, in our example one example is two-dimentional vector
# columns - training example, above we have 1000 training examples
Y = function_to_estimate.function1(X) # outputs

m = np.shape(X)[1] # number of training examples

# parameters initialization
for l in range(0,len(layers)):
    W[l] = np.random.randn(layers[l], layers[l-1]) * 0.01 # or / np.sqrt(layers[l])
    b[l] = np.zeros((layers[l], 1))

data = generate_data(0, 1000, 0.5, 0, 1000, 0.5)
data_indexes = np.random.choice([0, 1, 2], p = [0.8, 0.1, 0.1], size = (1, data.shape[1]))
training_data = data[:, (data_indexes == 0)[0]]
testing_data = data[:, (data_indexes == 1)[0]]
validation_data = data[:, (data_indexes == 2)[0]]
trX = training_data[0:2,:]
trY = np.reshape(training_data[2,:], (-1, 1))

#training
for i in range(0, number_of_iterations):
    #forward propagation through all the layers
    A, Z = forward(X, L, W, b, relu)

    cost = calculate_cost(A[L], Y)

    print(cost)

    #back propagation through all the layers
    # dA = np.multiply(np.transpose(W[L - 1]), (A - Y)) # not sure - check if correct
    dA = A[L] - Y #initialization (cost derivative)

    #Lth (last) layer:
    prevA = A[L - 1]
    dZ = dA
    prev_dA = np.transpose(W[L]) * dZ
    dW = (1. / m) * np.dot(dZ, np.transpose(prevA))
    db = 1. / m * np.sum(dZ, axis = 1, keepdims = True)
    W[L], b[L] = update_parameters(learning_rate, dW, db, W[L], b[L])

    #from (L-1)th to 1st layer:
    for l in reversed(range(1, L)):
        prevA = A[l - 1]
        dW, db, prev_dA = backprop(Z[l], prev_dA, prevA, W[l], m)
        W[l], b[l] = update_parameters(learning_rate, dW, db, W[l], b[l])



# print(W)
# print(b)

