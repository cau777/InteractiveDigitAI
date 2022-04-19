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
        layers_inputs[i + 1] = layers[i].feed_forward(layers_inputs[i])

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
            outputs = layer.feed_forward(inputs)
            loss_grad = loss_func.calc_loss_gradient(expected, outputs)
            layer.backpropagate_gradient(inputs, outputs, loss_grad, config)
            layer.train(config)

        final_loss = loss_func.calc_loss(expected, layer.feed_forward(inputs))
        self.assertLess(final_loss.mean(), 0.01)

    def test_training_3_layers(self):
        inputs = np.random.rand(1, 10)
        expected = np.random.rand(1, 10)

        loss_func = MseLossFunction()
        layers = [
            DenseLayer.create_random(10, 64, AdamLrOptimizer(), AdamLrOptimizer()),
            DenseLayer.create_random(64, 64, AdamLrOptimizer(), AdamLrOptimizer()),
            DenseLayer.create_random(64, 10, AdamLrOptimizer(), AdamLrOptimizer()),
        ]

        for e in range(75):
            config = TrainingConfig(e + 1, 1)

            layers_inputs = layers_feed_forward(layers, inputs)

            loss_grad = loss_func.calc_loss_gradient(expected, layers_inputs[-1])
            for i in range(len(layers) - 1, -1, -1):
                loss_grad = layers[i].backpropagate_gradient(layers_inputs[i], layers_inputs[i + 1], loss_grad, config)

            for i in layers:
                i.train(config)

        final_loss = loss_func.calc_loss(expected, layers_feed_forward(layers, inputs)[-1])
        self.assertLess(final_loss.mean(), 0.0001)


if __name__ == '__main__':
    unittest.main()
