import unittest

import numpy as np

from codebase.nn.loss_functions.mse_loss_function import MseLossFunction
from codebase.nn.lr_optimizers.adam_optimizer import AdamLrOptimizer
from codebase.nn.layers.dense_layer import DenseLayer
from codebase.nn.utils import init_random
from codebase.nn.training_config import TrainingConfig


def layers_feed_forward(layers: list[DenseLayer], inputs: np.ndarray):
    layers_inputs = list([None] * (len(layers) + 1))
    layers_inputs[0] = inputs

    for i in range(len(layers)):
        layers_inputs[i + 1] = layers[i].forward(layers_inputs[i])

    return layers_inputs


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        init_random()

    def test_training(self):
        inputs = np.random.rand(1, 10)
        expected = np.random.rand(1, 10)

        loss_func = MseLossFunction()
        layer = DenseLayer.create_random(10, 10, AdamLrOptimizer(), AdamLrOptimizer())

        for e in range(200):
            config = TrainingConfig(e + 1, 1)
            outputs, cache = layer.forward(inputs)
            loss_grad = loss_func.calc_loss_gradient(expected, outputs)
            layer.backward(loss_grad, cache)
            layer.train(config)

        final_loss = loss_func.calc_loss(expected, layer.forward(inputs)[0])
        self.assertLess(final_loss.mean(), 0.01)


if __name__ == '__main__':
    unittest.main()
