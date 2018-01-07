import numpy as np

import function_to_estimate
from evaluate import print_metrics, sanity_check
from sigmoid import sigmoid, sigmoid_backward
from leaky_relu import leaky_relu, leaky_relu_backward
from relu import relu, relu_backward

layers = [2, 2, 3, 3, 5, 5, 3, 3, 2, 2, 1]
L = len(layers) - 1  # number of layers - input layer doesn't count

W={}
b={}

W[1] = np.array([[ 0.01624345, -0.00611756],
 [-0.00528172, -0.01072969]])

W[2] = np.array([[ 0.00865408, -0.02301539],
 [ 0.01744812, -0.00761207],
 [ 0.00319039, -0.0024937 ]])

W[3] = np.array([[ 0.01462108, -0.02060141, -0.00322417],
 [-0.00384054,  0.01133769, -0.01099891],
 [-0.00172428, -0.00877858,  0.00042214]])

W[4] = np.array([[ 0.00582815, -0.01100619,  0.01144724],
 [ 0.00901591,  0.00502494,  0.00900856],
 [-0.00683728, -0.0012289 , -0.00935769],
 [-0.00267888,  0.00530355, -0.00691661],
 [-0.00396754, -0.00687173, -0.00845206]])

W[5] = np.array([[-0.00671246, -0.00012665, -0.0111731 ,  0.00234416,  0.01659802],
 [ 0.00742044, -0.00191836, -0.00887629, -0.00747158,  0.01692455],
 [ 0.00050808, -0.00636996,  0.00190915,  0.02100255,  0.00120159],
 [ 0.00617203,  0.0030017 , -0.0035225 , -0.01142518, -0.00349343],
 [-0.00208894,  0.00586623,  0.00838983,  0.00931102,  0.00285587]])

W[6] = np.array([[ 0.00885141, -0.00754398,  0.01252868,  0.0051293 , -0.00298093],
 [ 0.00488518, -0.00075572,  0.01131629,  0.01519817,  0.02185575],
 [-0.01396496, -0.01444114, -0.00504466,  0.00160037,  0.00876169]])

W[7] = np.array([[ 0.00315635, -0.02022201, -0.00306204],
 [ 0.00827975,  0.00230095,  0.00762011],
 [-0.00222328, -0.00200758,  0.00186561]])

W[8] = np.array([[ 0.00410052,  0.001983  ,  0.00119009],
 [-0.00670662,  0.00377564,  0.00121821]])

W[9] = np.array([[ 0.01129484,  0.01198918],
 [ 0.00185156, -0.00375285]])

W[10] = np.array([[-0.0064571 ,  0.00423494]])

b[1] = np.array([[  2.71049802e-15],
 [ -1.48885142e-15]])

b[2] = np.array([[  4.54733957e-19],
 [  6.28730078e-13],
 [ -1.20004229e-13]])

b[3] = np.array([[  0.00000000e+00],
 [  7.08221599e-17],
 [  0.00000000e+00]])

b[4] = np.array([[  0.00000000e+00],
 [ -3.13677984e-11],
 [  0.00000000e+00],
 [  4.69139977e-14],
 [  0.00000000e+00]])

b[5] = np.array([[ -4.24709690e-10],
 [  0.00000000e+00],
 [  5.93464432e-12],
 [  0.00000000e+00],
 [  6.00228882e-12]])

b[6] = np.array([[  5.38010158e-10],
 [  1.49513401e-10],
 [  4.95147719e-10]])

b[7] = np.array([[  0.00000000e+00],
 [  6.49790622e-08],
 [  0.00000000e+00]])

b[8] = np.array([[  1.08465856e-05],
 [  1.15133694e-05]])

b[9] = np.array([[ 0.00096031],
 [ 0.        ]])

b[10] = np.array([[-0.14483269]])

activation_fun = {'forward': relu, 'backward': relu_backward}

sanity_check(L, W, b, activation_fun, function_to_estimate.function1)
print_metrics(layers, W, b, activation_fun)
