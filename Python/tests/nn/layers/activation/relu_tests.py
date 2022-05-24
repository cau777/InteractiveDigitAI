import unittest
import numpy as np

from codebase.nn import BatchConfig
from codebase.nn.layers.activation import ReluLayer


def relu(x):
    return max(0, x)


def relu_gradient(x):
    return 1 if x > 0 else 0


class MyTestCase(unittest.TestCase):
    def test_forward(self):
        arr = np.random.rand(10, 10)
        expected = np.vectorize(relu)(arr)
        layer = ReluLayer()
        result, _ = layer.forward(arr, BatchConfig(False))
        self.assertTrue(np.array_equal(expected, result))

    def test_backprop(self):
        arr = np.random.rand(10, 10)
        grad = np.random.rand(10, 10)
        expected = grad * np.vectorize(relu_gradient)(arr)
        layer = ReluLayer()

        config = BatchConfig(True)
        output, cache = layer.forward(arr, config)
        self.assertTrue(np.array_equal(expected, layer.backward(grad, cache, config)))


if __name__ == '__main__':
    unittest.main()
