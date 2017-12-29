import unittest
import numpy as np
from relu import relu
from forward import forward
from sigmoid import sigmoid

def linear(x):
    return x

class TestForward(unittest.TestCase):

    def test_forward_linear_1(self):
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

    def test_forward_linear_2(self):
        X = np.array([[-1.2], [3.5]])
        W = {}
        b = {}
        L = 3
        W[1] = np.array([[0.1, 1.7], [8.2, -0.4], [-5.5, 1.7]])
        b[1] = np.array([[-3.2], [0.2], [-1.9]])
        W[2] = np.array([[0.5, 1.7, -1.1], [-0.4, -0.3, 1.1], [7.1, 3.3, -4.7]])
        b[2] = np.array([[1.0], [3.9], [-3.0]])
        W[3] = np.array([[-2.2, 2.7, 0.1]])
        b[3] = np.array([[-0.4]])
        A, _ = forward(X, L, W, b, linear)
        result = np.all(np.isclose(A[L], np.array([[102.7507]])))
        self.assertTrue(result)

    def test_forward_relu_1(self):
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

    def test_forward_relu_2(self):
        X = np.array([[-1.2], [3.5]])
        W = {}
        b = {}
        L = 3
        W[1] = np.array([[0.1, 1.7], [8.2, -0.4], [-5.5, 1.7]])
        b[1] = np.array([[-3.2], [0.2], [-1.9]])
        W[2] = np.array([[0.5, 1.7, -1.1], [-0.4, -0.3, 1.1], [7.1, 3.3, -4.7]])
        b[2] = np.array([[1.0], [3.9], [-3.0]])
        W[3] = np.array([[-2.2, 2.7, 0.1]])
        b[3] = np.array([[-0.4]])
        A, _ = forward(X, L, W, b, relu)
        result = np.all(np.isclose(A[L], np.array([[38.9201]])))
        self.assertTrue(result)

    def test_forward_sigmoid_1(self):
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

    def test_forward_sigmoid_2(self):
        X = np.array([[-1.2], [3.5]])
        W = {}
        b = {}
        L = 3
        W[1] = np.array([[0.1, 1.7], [8.2, -0.4], [-5.5, 1.7]])
        b[1] = np.array([[-3.2], [0.2], [-1.9]])
        W[2] = np.array([[0.5, 1.7, -1.1], [-0.4, -0.3, 1.1], [7.1, 3.3, -4.7]])
        b[2] = np.array([[1.0], [3.9], [-3.0]])
        W[3] = np.array([[-2.2, 2.7, 0.1]])
        b[3] = np.array([[-0.4]])
        A, _ = forward(X, L, W, b, sigmoid)
        result = np.all(np.isclose(A[L], np.array([[0.999924175259053]])))
        self.assertTrue(result)

    def linear_function(x):
        return x

if __name__ == '__main__':
    unittest.main()
