import unittest
import numpy as np
from relu import relu
from forward import forward
from sigmoid import sigmoid

def linear(x):
    return x

class TestForward(unittest.TestCase):

    def test_forward_linear(self):
        X = np.array([[3.2], [5.8]])
        W = {}
        b = {}
        L = 2
        W[1] = np.array([[1.3, -2.7]])
        b[1] = np.array([[-0.2]])
        W[2] = np.array([[3.1]])
        b[2] = np.array([[-0.9]])
        A, _ = forward(X, L, W, b, linear)
        result = np.all(np.isclose(A[L], np.array([[-37.17]])))
        self.assertTrue(result)

    def test_forward_relu(self):
        X = np.array([[3.2], [5.8]])
        W = {}
        b = {}
        L = 2
        W[1] = np.array([[1.3, -2.7]])
        b[1] = np.array([[-0.2]])
        W[2] = np.array([[3.1]])
        b[2] = np.array([[-0.9]])
        A, _ = forward(X, L, W, b, relu)
        result = np.all(np.isclose(A[L], np.array([[-0.9]])))
        self.assertTrue(result)

    def test_forward_sigmoid(self):
        X = np.array([[3.2], [5.8]])
        W = {}
        b = {}
        L = 2
        W[1] = np.array([[1.3, -2.7]])
        b[1] = np.array([[-0.2]])
        W[2] = np.array([[3.1]])
        b[2] = np.array([[-0.9]])
        A, _ = forward(X, L, W, b, sigmoid)
        result = np.all(np.isclose(A[L], np.array([[-0.899974289]])))
        self.assertTrue(result)

    def linear_function(x):
        return x

if __name__ == '__main__':
    unittest.main()
