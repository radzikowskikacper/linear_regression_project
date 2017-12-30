import numpy as np
from scipy.stats import pearsonr, spearmanr

from data_gen import generate_data_sets
from forward import forward

def evaluate_model(layers, W, b, activation_fun):
    L = len(layers) - 1  # number of layers - input layer doesn't count

    train_data_min = -20
    train_data_max = 20
    train_data_step = 1
    test_data_min = -20
    test_data_max = 20
    test_data_step = 1
    trX, trY, tsX, tsY, validation_data = generate_data_sets(train_data_min, train_data_max, train_data_step,
                                                             test_data_min, test_data_max, test_data_step)

    tsA, _ = forward(tsX, L, W, b, activation_fun['forward'])
    print("Regression metrics:")
    print("MSE:", np.power(tsA[L] - tsY, 2).mean() / np.shape(tsA[3])[1])
    print("R2: ", np.sum(np.power(tsA[L] - tsY.mean(), 2)) / np.sum(np.power(tsY - tsY.mean(), 2)))
    pearson, pearson_pvalue = pearsonr(tsA[L].reshape(tsA[L].shape[1]), tsY.reshape(tsY.shape[1]))
    spearman, spearman_pvalue = spearmanr(tsA[L].reshape(tsA[L].shape[1]), tsY.reshape(tsY.shape[1]))
    print("Pearson: {:.3f}, p-value: {:.3f}".format(pearson, pearson_pvalue))
    print("Spearman: {:.3f}, p-value: {:.3f}".format(spearman, spearman_pvalue))
