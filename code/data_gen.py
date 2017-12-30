import numpy as np
import itertools as ittools

from function_to_estimate import function1, function2

def generate_data_sets(train_data_min, train_data_max, train_data_step, test_data_min, test_data_max, test_data_step):
    np.random.seed(1)
    data = generate_data_helper(train_data_min, train_data_max, train_data_step, test_data_min, test_data_max, test_data_step)
    data_indexes = np.random.choice([0, 1, 2], p=[0.8, 0.1, 0.1], size=(1, data.shape[1]))
    training_data = data[:, (data_indexes == 0)[0]]
    testing_data = data[:, (data_indexes == 1)[0]]
    validation_data = data[:, (data_indexes == 2)[0]]
    trX = training_data[0:2, :]
    trY = np.reshape(training_data[2, :], (1, -1))
    tsX = testing_data[0:2, :]
    tsY = np.reshape(testing_data[2, :], (1, -1))
    return trX, trY, tsX, tsY, validation_data

def generate_data_helper(x1min, x1max, x1step, x2min, x2max, x2step):
    np.random.seed(1)
    arr = np.array(list(ittools.product(np.arange(x1min, x1max, x1step), np.arange(x2min, x2max, x2step))))
    arr = np.reshape(arr.T, (2, -1))
    arr = np.vstack((arr, function1(arr), function2(arr)))
    return arr