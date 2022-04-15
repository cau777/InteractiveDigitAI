import unittest

import numpy as np

from nn import TrainingConfig
from nn.loss_functions.cross_entropy_loss_function import softmax
from nn.loss_functions import CrossEntropyLossFunction
from nn.layers import DenseLayer
from nn.lr_optimizers import ConstantLrOptimizer
from nn.utils import init_random


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        init_random()

    def test_softmax(self):
        arr = np.random.rand(10, 1)
        soft = softmax(arr)
        self.assertGreater(0.0001, abs(1 - np.sum(soft)))

    def test_training(self):
        inputs = np.random.rand(100, 1)
        expected = np.zeros((10, 1))
        expected[0] = 1

        loss_func = CrossEntropyLossFunction()
        layer = DenseLayer.create_random(100, 10, ConstantLrOptimizer(), ConstantLrOptimizer())
        prev_loss = None

        for e in range(50):
            config = TrainingConfig(e + 1, 1)
            outputs = layer.feed_forward(inputs)
            loss = loss_func.calc_loss(expected, outputs)
            loss_grad = loss_func.calc_loss_gradient(expected, outputs)
            layer.backpropagate_gradient(inputs, outputs, loss_grad, config)
            layer.train(config)

            if prev_loss is not None:
                self.assertLess(loss, prev_loss)
            prev_loss = loss


if __name__ == '__main__':
    unittest.main()
