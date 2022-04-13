import unittest

import numpy as np

from nn.layers import convolution_layer
from nn.loss_functions.mse_loss_function import MseLossFunction
from nn.lr_optimizers.adam_optimizer import AdamLrOptimizer
from nn.training_config import TrainingConfig
from nn.utils import init_random


def get_image_example():
    return (
        np.array([[
            [3, 0, 1, 2, 7, 4],
            [1, 5, 8, 9, 3, 1],
            [2, 7, 2, 5, 1, 3],
            [0, 1, 3, 1, 7, 8],
            [4, 2, 1, 6, 2, 8],
            [2, 4, 5, 2, 3, 9],
        ]]),
        np.array([[
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1],
        ]]),
        np.array([[
            [-5, -4, 0, 8],
            [-10, -2, 2, 3],
            [0, -2, -4, -7],
            [-3, -2, -3, -16],
        ]])
    )


class ConvolutionLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        init_random()

    def test_padding0(self):
        array = np.random.rand(1, 8, 8)
        output = convolution_layer.pad3d(array, 0)
        self.assertTrue(np.array_equal(array, output))

    def test_padding(self):
        array = np.random.rand(1, 8, 8)

        for p in range(1, 7):
            with self.subTest(i=p):
                output = convolution_layer.pad3d(array, p)
                expected = np.zeros((1, 8 + 2 * p, 8 + 2 * p))

                for h in range(p, 8 + p):
                    for w in range(p, 8 + p):
                        expected[0, h, w] = array[0, h - p, w - p]

                self.assertEqual(expected.shape, output.shape)
                self.assertTrue(np.array_equal(expected, output))

    def test_feed_forward(self):
        image, kernel, expected = get_image_example()

        layer = convolution_layer.ConvolutionLayer(kernel, None)
        output = layer.feed_forward(image)
        self.assertEqual(expected.shape, output.shape)
        self.assertTrue(np.array_equal(expected, output))

    def test_training(self):
        image, _, expected = get_image_example()
        image = image * 0.1
        expected = expected * 0.1

        layer = convolution_layer.build_random(1, 1, 3, AdamLrOptimizer(0.05))
        loss_func = MseLossFunction()

        for e in range(100):
            output = layer.feed_forward(image)
            config = TrainingConfig(e+1, 1)
            loss_grad = loss_func.calc_loss_gradient(expected, output)
            layer.backpropagate_gradient(image, output, loss_grad, config)
            layer.train(config)

        final_loss = loss_func.calc_loss(expected, layer.feed_forward(image))
        self.assertLess(final_loss.mean(), 0.01)


if __name__ == '__main__':
    unittest.main()
