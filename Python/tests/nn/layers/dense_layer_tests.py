import unittest

import numpy as np

from codebase.nn.loss_functions.mse_loss_function import MseLossFunction
from codebase.nn.lr_optimizers.adam_optimizer import AdamLrOptimizer
from codebase.nn.layers.dense_layer import DenseLayer
from codebase.nn.utils import init_random
from codebase.nn.training_config import BatchConfig


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        init_random()

    def test_training(self):
        inputs = np.random.rand(1, 10).astype("float32")
        expected = np.random.rand(1, 10).astype("float32")

        loss_func = MseLossFunction()
        layer = DenseLayer.create_random(10, 10, AdamLrOptimizer(), AdamLrOptimizer())

        for e in range(200):
            config = BatchConfig(True, e + 1)
            outputs, cache = layer.forward(inputs, config)
            loss_grad = loss_func.calc_loss_gradient(expected, outputs)
            self.assertEqual(outputs.dtype, "float32")
            grad = layer.backward(loss_grad, cache, config)
            self.assertEqual(grad.dtype, "float32")
            layer.train(config)

        final, _ = layer.forward(inputs, BatchConfig(False, 200))
        final_loss = loss_func.calc_loss(expected, final)
        self.assertLess(final_loss.mean(), 0.01)


if __name__ == '__main__':
    unittest.main()
