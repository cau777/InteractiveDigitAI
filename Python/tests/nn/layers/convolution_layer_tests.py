import unittest
import numpy as np

from codebase.nn import TrainingConfig
from codebase.nn.layers import ConvolutionLayer
from codebase.nn.layers import convolution_layer
from codebase.nn.loss_functions import MseLossFunction
from codebase.nn.lr_optimizers import AdamLrOptimizer
from codebase.nn.utils import init_random
from tests.profiling_utils import profiler


def get_image_example():
    return (
        np.array([[[
            [3, 0, 1, 2, 7, 4],
            [1, 5, 8, 9, 3, 1],
            [2, 7, 2, 5, 1, 3],
            [0, 1, 3, 1, 7, 8],
            [4, 2, 1, 6, 2, 8],
            [2, 4, 5, 2, 3, 9],
        ]]]),
        np.array([[
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1],
        ]]),
        np.array([[[
            [-5, -4, 0, 8],
            [-10, -2, 2, 3],
            [0, -2, -4, -7],
            [-3, -2, -3, -16],
        ]]])
    )


class ConvolutionLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        init_random()

    @classmethod
    def tearDownClass(cls):
        profiler.print_stats()

    def test_padding0(self):
        array = np.random.rand(1, 1, 8, 8)
        output = convolution_layer.pad4d(array, 0)
        self.assertTrue(np.array_equal(array, output))

    def test_padding(self):
        array = np.random.rand(1, 1, 8, 8)

        for p in range(1, 7):
            with self.subTest(i=p):
                output = convolution_layer.pad4d(array, p)
                expected = np.zeros((1, 1, 8 + 2 * p, 8 + 2 * p))

                for h in range(p, 8 + p):
                    for w in range(p, 8 + p):
                        expected[0, 0, h, w] = array[0, 0, h - p, w - p]

                self.assertEqual(expected.shape, output.shape)
                self.assertTrue(np.array_equal(expected, output))

    def test_feed_forward(self):
        image, kernel, expected = get_image_example()

        layer = convolution_layer.ConvolutionLayer(kernel, None)
        output, _ = layer.forward(image)
        self.assertEqual(expected.shape, output.shape)
        self.assertTrue(np.array_equal(expected, output))

    def test_feed_forward_multiple(self):
        image, kernel, expected = get_image_example()
        image_batch = np.vstack([image, -image])
        expected_batch = np.vstack([expected, -expected])

        layer = convolution_layer.ConvolutionLayer(kernel, None)
        output, _ = layer.forward(image_batch)
        self.assertEqual(expected_batch.shape, output.shape)
        self.assertTrue(np.array_equal(expected_batch, output))

    def test_feed_forward_multiple_with_channels(self):
        batch_size = 2
        in_channels = 3
        out_channels = 4
        height = 10
        width = 15
        kernel_size = 5

        image = np.random.rand(batch_size, in_channels, height, width)
        kernel = np.random.rand(out_channels, in_channels, kernel_size, kernel_size)

        layer = convolution_layer.ConvolutionLayer(kernel, None)
        output, cache = layer.forward(image)
        expected_output_shape = (batch_size, out_channels, height - kernel_size + 1, width - kernel_size + 1)

        grad = layer.backward(output, cache)
        expected_grad_shape = (batch_size, in_channels, height, width)
        self.assertEqual(expected_output_shape, output.shape)
        self.assertEqual(expected_grad_shape, grad.shape)

    def test_training_with_example(self):
        image, _, expected = get_image_example()
        image = image * 0.1
        expected = expected * 0.1

        layer = ConvolutionLayer.create_random(1, 1, 3, AdamLrOptimizer(0.05))
        loss_func = MseLossFunction()

        for e in range(100):
            inputs = np.vstack([image, image])
            output, cache = layer.forward(inputs)
            config = TrainingConfig(e + 1, 2)
            loss_grad = loss_func.calc_loss_gradient(expected, output)
            layer.backward(loss_grad, cache)
            layer.train(config)

        final_loss = loss_func.calc_loss(expected, layer.forward(image)[0])
        self.assertLess(final_loss.mean(), 0.0001)

    def test_training(self):
        image = np.abs(np.random.rand(10, 7, 5, 5))
        expected = np.random.rand(10, 4, 3, 3)

        layer = ConvolutionLayer.create_random(7, 4, 3, AdamLrOptimizer(0.05))
        loss_func = MseLossFunction()

        for e in range(100):
            output, cache = layer.forward(image)
            config = TrainingConfig(e + 1, 2)
            loss_grad = loss_func.calc_loss_gradient(expected, output)
            layer.backward(loss_grad, cache)
            layer.train(config)

            # final_loss = loss_func.calc_loss(expected, layer.feed_forward(image))
            # print(final_loss.mean())

        final_loss = loss_func.calc_loss(expected, layer.forward(image)[0])
        self.assertLess(final_loss.mean(), 0.3)


if __name__ == '__main__':
    unittest.main()
