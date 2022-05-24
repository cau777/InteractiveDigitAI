import unittest

import numpy as np

from codebase.nn import TrainingConfig
from codebase.nn.loss_functions.cross_entropy_loss_function import softmax
from codebase.nn.loss_functions import CrossEntropyLossFunction
from codebase.nn.layers import DenseLayer
from codebase.nn.lr_optimizers import ConstantLrOptimizer
from codebase.nn.utils import init_random


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        init_random()

    def test_softmax(self):
        arr = np.random.rand(1, 10)
        soft = softmax(arr)
        self.assertGreater(0.0001, abs(1 - np.sum(soft)))

    def test_softmax_batch(self):
        arr = np.random.rand(16, 10)
        soft = softmax(arr)
        self.assertGreater(0.0001, abs(16 - np.sum(soft)))

    def test_error_batch(self):
        arr = np.abs(np.random.rand(16, 10))
        layer = CrossEntropyLossFunction()
        loss = layer.calc_loss(arr, arr * 0.5)
        self.assertEqual((16,), loss.shape)

    def test_error_small_batch(self):
        arr = np.abs(np.random.rand(8, 10))
        layer = CrossEntropyLossFunction()
        loss = layer.calc_loss(arr, arr * 0.5)
        self.assertEqual((8,), loss.shape)

    def test_training(self):
        inputs = np.random.rand(1, 100)
        expected = np.zeros((1, 10))
        expected[0] = 1

        loss_func = CrossEntropyLossFunction()
        layer = DenseLayer.create_random(100, 10, ConstantLrOptimizer(), ConstantLrOptimizer())
        prev_loss = None

        for e in range(50):
            config = TrainingConfig(e + 1, 1)
            outputs, cache = layer.forward(inputs)
            loss = loss_func.calc_loss(expected, outputs)
            loss_grad = loss_func.calc_loss_gradient(expected, outputs)
            layer.backward(loss_grad, cache)
            layer.train(config)

            if prev_loss is not None:
                self.assertLess(loss, prev_loss)
            prev_loss = loss


if __name__ == '__main__':
    unittest.main()
