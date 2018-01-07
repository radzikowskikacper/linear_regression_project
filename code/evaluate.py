import numpy as np
from scipy.stats import pearsonr, spearmanr

from data_gen import generate_data_sets, generate_data_helper
from forward import forward

def print_metrics(layers, W, b, activation_fun):
    L = len(layers) - 1  # number of layers - input layer doesn't count

    train_data_min = -20
    train_data_max = 20
    train_data_step = 1
    test_data_min = -20
    test_data_max = 20
    test_data_step = 1
    trX, trY, tsX, tsY, validation_data = generate_data_sets(generate_data_helper(train_data_min, train_data_max, train_data_step,
                                                             test_data_min, test_data_max, test_data_step))

    tsA, _ = forward(tsX, L, W, b, activation_fun['forward'])
    print("Regression metrics:")
    print("MSE:", np.power(tsA[L] - tsY, 2).mean() / np.shape(tsA[L])[1])
    # print("R2: ", np.sum(np.power(tsA[L] - tsY.mean(), 2)) / np.sum(np.power(tsY - tsY.mean(), 2)))
    SSreg = np.sum(np.power(tsA[L] - tsY.mean(), 2))
    SSres = np.sum(np.power(tsA[L] - tsY, 2))
    # SStot = np.sum(np.power(tsY - tsY.mean(), 2))
    SStot = SSres + SSreg
    R2 = 1 - SSres / SStot
    print("R2: ", R2)
    pearson, pearson_pvalue = pearsonr(tsA[L].reshape(tsA[L].shape[1]), tsY.reshape(tsY.shape[1]))
    spearman, spearman_pvalue = spearmanr(tsA[L].reshape(tsA[L].shape[1]), tsY.reshape(tsY.shape[1]))
    print("Pearson: {:.3f}, p-value: {:.3f}".format(pearson, pearson_pvalue))
    print("Spearman: {:.3f}, p-value: {:.3f}".format(spearman, spearman_pvalue))

def sanity_check(L, W, b, activation_fun, function_to_estimate):
    def predict(X):
        A, _ = forward(X, L, W, b, activation_fun['forward'])
        Y = function_to_estimate(X)
        print("X={}".format(np.array2string(X, separator=', ')))
        print("real value: ", Y)
        print("prediction: {}\n".format(A[L]))

    # sanity check
    print("Sanity check:")
    np.random.seed(1)
    predict(np.random.rand(2, 1) * [[20], [20]])
    predict(np.random.rand(2, 1) * [[-20], [20]])
    predict(np.random.rand(2, 1) * [[20], [-20]])
    predict(np.random.rand(2, 1) * [[-20], [-20]])