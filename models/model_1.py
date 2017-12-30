import numpy as np

from evaluate import evaluate_model
from sigmoid import sigmoid, sigmoid_backward

layers = [2, 10, 10, 1]

W =  {1: np.array([[  5.50749231e+00,  -1.08579520e+00],
       [  9.93049612e-01,  -6.81820355e+00],
       [ -6.30553767e+00,  -2.54175870e+00],
       [ -5.00538067e+00,  -9.60277867e-01],
       [  1.94265150e-01,   4.12717914e+00],
       [  5.02548904e-01,   5.46438221e+00],
       [ -1.96983067e+00,  -8.24143303e+00],
       [ -3.49553448e+00,   4.49910203e-03],
       [  5.80475640e-02,   4.45876046e+00],
       [ -4.04438565e+00,  -5.87095451e-01]]), 2: np.array([[ 0.0618346 ,  1.51550481, -2.47763666,  0.88012553,  2.07307459,
         2.02109013,  4.54990139,  1.88108217,  1.56468537,  0.97909935],
       [ 0.52673565,  0.50354985, -3.83023353,  0.93073336,  0.88226385,
         0.92279662,  3.3056987 ,  1.46991147,  0.48997441,  0.74286291],
       [ 0.15805644,  1.49386675, -2.44770813,  1.03220862,  2.07221395,
         2.01529507,  4.82008102,  1.70034804,  1.57103305,  1.08756702],
       [ 0.59269983,  0.53134506, -3.81801554,  0.6702609 ,  0.85492131,
         0.916011  ,  4.08139042,  1.455408  ,  0.41347014,  0.50877125],
       [ 5.90920058,  2.01092017, -4.33343039, -0.1568687 ,  0.78419501,
         0.89144281,  5.54896497,  1.80574168,  0.27887805, -0.18224904],
       [ 0.61492654,  0.61913982, -3.84532312,  0.62597496,  0.84374119,
         0.8780143 ,  4.31023133,  1.51571126,  0.3769419 ,  0.45838863],
       [ 0.18189175,  1.51189639, -2.41952349,  1.09021955,  2.05596475,
         2.01528438,  4.75522179,  1.64155714,  1.55999481,  1.11797862],
       [ 0.93796584,  1.11215037, -2.4733048 ,  0.3916981 ,  2.06603414,
         2.08101971,  2.44154622,  2.66988967,  1.33350161,  0.60384267],
       [ 0.72709073,  1.11782351, -2.4620862 ,  0.42208518,  2.05712611,
         2.05472486,  2.77083931,  2.55189605,  1.35585638,  0.63080906],
       [ 0.56763773,  0.5657231 , -3.82726915,  0.70938704,  0.83646761,
         0.90910958,  3.97946637,  1.48480169,  0.41864323,  0.53212013]]), 3: np.array([[-4.67968546, -4.35454081, -4.77845039, -4.49593602, -6.81689482,
        -4.56219656, -4.75249375, -4.44251859, -4.32852246, -4.46324813]])}


b = {1: np.array([[ 21.98742748],
       [ 17.91995317],
       [ -1.10102039],
       [ -1.31284867],
       [ -4.06765371],
       [ -2.63457872],
       [  1.35453304],
       [-10.03013863],
       [ -2.43389125],
       [ -4.1647291 ]]), 2: np.array([[-6.20519618],
       [-5.70058717],
       [-6.45738936],
       [-5.68249058],
       [-6.96826769],
       [-5.70100966],
       [-6.47248996],
       [-5.00803982],
       [-5.04862898],
       [-5.67143111]]), 3: np.array([[ 3.27478309]])}

activation_fun = {'forward': sigmoid, 'backward': sigmoid_backward}

evaluate_model(layers, W, b, activation_fun)