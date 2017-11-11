import unittest
import numpy as np
from relu import relu, backward

class TestRelu(unittest.TestCase):

    def test_relu(self):
        X = np.array([0, 3.54531542, -4.81741295, -2.73031028, -16.23282883, 4.96819628, 3.85990703, -16.42697689, -2.91987025, -3.20573417, 8.31951293])
        expected = np.array([0, 3.54531542, 0, 0, 0, 4.96819628, 3.85990703, 0, 0, 0, 8.31951293])
        Y = relu(X)
        result = np.all(np.isclose(Y, expected))
        self.assertTrue(result)

    def test_relu_backward(self):
        dA = np.array([0, 3.54531542, -4.81741295, -2.73031028, -16.23282883, 4.96819628, 3.85990703, -16.42697689, -2.91987025, -3.20573417, 8.31951293])
        Z = np.array([-1.25541855, 1.15025199, 0.64849487, 0.82883564, 0.11937011, 0.2328452, 2.40720121, -0.14718964, 0.92134979, -1.48488645, -0.630742])
        expected = np.array([0, 3.54531542, -4.81741295, -2.73031028, -16.23282883, 4.96819628, 3.85990703, 0, -2.91987025, 0, 0])
        Y = backward(dA, Z)
        result = np.all(np.isclose(Y, expected))
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
