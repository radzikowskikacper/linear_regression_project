import numpy as np

layers = [2, 5, 4, 1] # number of units in each layer
L = len(layers) # number of layer
W = {}
b = {}
number_of_iterations = 100
learning_rate = 0.1

np.random.seed(100) # to have the same results (testing) - remove in final solution

# parameters initialization
for l in range(1,len(layers)):
    W[l] = np.random.randn(layers[l], layers[l-1]) / np.sqrt(layers[l])
    b[l] = np.zeros((layers[l], 1))

for i in range(0, number_of_iterations):
    pass

print(W)
print(b)

