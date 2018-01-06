import os
import numpy as np

def frange(a, z):
    return range(a, z + 1)

def save_model(layers, W, b, activation_fun, tr_hist, number_of_iterations, learning_rate, function_to_estimate, save_cost_plot_after_epoch):
    L = len(layers) - 1
    max_model_num = 0
    for file_name in os.listdir("./../models/"):
        if (file_name.startswith("model_") and file_name.endswith(".py")):
            model_num = int(file_name[6:-3])
            if (model_num > max_model_num):
                max_model_num = model_num

    model_file = open('./../models/model_{}.py'.format(max_model_num+1), 'w')
    model_file.write("import numpy as np\n\n")
    model_file.write("import function_to_estimate\n")
    model_file.write("from evaluate import print_metrics, sanity_check\n")
    model_file.write("from sigmoid import sigmoid, sigmoid_backward\n")
    model_file.write("from leaky_relu import leaky_relu, leaky_relu_backward\n")
    model_file.write("from relu import relu, relu_backward\n\n")
    model_file.write("layers = {}\n".format(layers))
    model_file.write("L = len(layers) - 1  # number of layers - input layer doesn't count\n\n")
    model_file.write("W={}\n")
    model_file.write("b={}\n\n")

    np.set_printoptions(threshold=np.inf)
    for l in frange(1,L):
        model_file.write("W[{}] = np.array({})\n\n".format(l, np.array2string(W[l], separator=', ')))
    for l in frange(1, L):
        model_file.write("b[{}] = np.array({})\n\n".format(l, np.array2string(b[l], separator=', ')))

    activation_fun_name = str(activation_fun['forward']).split(' ')[1]
    model_file.write("activation_fun = {{'forward': {}, 'backward': {}_backward}}\n\n".format(activation_fun_name, activation_fun_name))
    model_file.write("sanity_check(L, W, b, activation_fun, function_to_estimate.{})\n".format(function_to_estimate.__name__))
    model_file.write("print_metrics(layers, W, b, activation_fun)\n")
    model_file.close()

    tr_hist_file = open('./../models/model_{}_train_hist'.format(max_model_num + 1), 'w')
    tr_hist_file.write("'''\n")
    tr_hist_file.write("Model generated with following hyperparams:\n")
    tr_hist_file.write("number_of_iterations = {}\n".format(number_of_iterations))
    tr_hist_file.write("learning_rate = {}\n\n".format(learning_rate))
    tr_hist_file.write("Train history:\n")
    tr_hist_file.write('\n'.join(tr_hist))
    tr_hist_file.write("\n'''")
    tr_hist_file.close()

    for epoch_plot in save_cost_plot_after_epoch:
        os.rename('./costs_after_{}_epoch.png'.format(epoch_plot), './../models/model_{}_costs_after_{}_epoch.png'.format(max_model_num + 1, epoch_plot))
    os.rename('./costs.png', './../models/model_{}_costs.png'.format(max_model_num + 1))

