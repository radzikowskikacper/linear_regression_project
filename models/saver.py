import os
import numpy as np

def frange(a, z):
    return range(a, z + 1)

def save_model(layers, W, b, activation_fun):
    L = len(layers) - 1
    max_model_num = 0
    for file_name in os.listdir("./../models/"):
        if (file_name.startswith("model_") and file_name.endswith(".py")):
            model_num = int(file_name[6:-3])
            if (model_num > max_model_num):
                max_model_num = model_num

    model_file = open('./../models/model_{}.py'.format(max_model_num+1), 'w')
    model_file.write("import numpy as np\n\n")
    model_file.write("from evaluate import print_metrics\n")
    model_file.write("from sigmoid import sigmoid, sigmoid_backward\n")
    model_file.write("from leaky_relu import leaky_relu, leaky_relu_backward\n")
    model_file.write("from relu import relu, relu_backward\n\n")
    model_file.write("layers = {}\n\n".format(layers))
    model_file.write("W={}\n")
    model_file.write("b={}\n\n")

    np.set_printoptions(threshold=np.inf)
    for l in frange(1,L):
        model_file.write("W[{}] = np.array({})\n\n".format(l, np.array2string(W[l], separator=', ')))
    for l in frange(1, L):
        model_file.write("b[{}] = np.array({})\n\n".format(l, np.array2string(b[l], separator=', ')))

    activation_fun_name = str(activation_fun['forward']).split(' ')[1]
    model_file.write("activation_fun = {{'forward': {}, 'backward': {}_backward}}\n\n".format(activation_fun_name, activation_fun_name))
    model_file.write("print_metrics(layers, W, b, activation_fun)\n")
