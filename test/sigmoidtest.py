import unittest
import numpy as np
from sigmoid import sigmoid, sigmoid_backward

class TestSigmoid(unittest.TestCase):

    def test_sigmoid(self):
        X = np.array([0, 0.19721045, 0.17785665, -0.23369414, 0.55528333, 2.07653441, -1.23360985, -0.84550586, -1.45364079, -0.61059541, 1.42725904])
        expected = np.array([0.5, 0.549143441907707, 0.544347320897941, 0.441840911014308, 0.635360493036452, 0.888601439999118, 0.225550242127031, 0.300376454902037, 0.189441877204342, 0.35192338881577, 0.806473883161429])
        Y = sigmoid(X)
        result = np.all(np.isclose(Y, expected))
        self.assertTrue(result)

    def test_backward(self):
        dA = np.array([0, 3.54531542, -4.81741295, -2.73031028, -16.23282883, 4.96819628, 3.85990703, -16.42697689, -2.91987025, -3.20573417, 8.31951293])
        Z = np.array([-1.25541855, 1.15025199, 0.64849487, 0.82883564, 0.11937011, 0.2328452, 2.40720121, -0.14718964, 0.92134979, -1.48488645, -0.630742])
        expected = np.array([0., 0.64748164, -1.08610566, -0.57757362, -4.0437849, 1.22536502, 0.29257434, -4.08458135, -0.59459819, -0.48271916, 1.88599722])
        Y = sigmoid_backward(dA, Z)
        result = np.all(np.isclose(Y, expected))
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
