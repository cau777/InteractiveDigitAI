import unittest

import numpy as np

from codebase.nn import BatchConfig
from codebase.nn.utils import init_random
from codebase.nn.layers.max_pool import MaxPoolLayer
from codebase.nn.utils import get_dims_after_filter


def manual_pooling(array: np.ndarray, size: int, stride: int):
    batch_size, channels, new_height, new_width = get_dims_after_filter(array.shape, size, stride)
    result = np.zeros((batch_size, channels, new_height, new_width), dtype="float32")

    for b in range(batch_size):
        for c in range(channels):
            for h in range(new_height):
                for w in range(new_width):
                    m = -10000
                    h_offset = h * stride
                    w_offset = w * stride

                    for i in range(size):
                        for j in range(size):
                            m = max(m, array[b, c, h_offset + i, w_offset + j])
                    result[b, c, h, w] = m
    return result


def manual_backprop(array: np.ndarray, grad: np.ndarray, size: int, stride: int):
    batch_size, channels, new_height, new_width = get_dims_after_filter(array.shape, size, stride)
    result = array * 0.0

    for b in range(batch_size):
        for c in range(channels):
            for h in range(new_height):
                for w in range(new_width):
                    m = -10000
                    h_offset = h * stride
                    w_offset = w * stride
                    best_h = 0
                    best_w = 0

                    for i in range(size):
                        for j in range(size):
                            val = array[b, c, h_offset + i, w_offset + j]
                            if val > m:
                                m = val
                                best_h = h_offset + i
                                best_w = w_offset + j
                    result[b, c, best_h, best_w] += grad[b, c, h, w]
    return result


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        init_random()

    def test_feed_forward0(self):
        array = np.random.rand(1, 1, 8, 8).astype("float32")
        layer = MaxPoolLayer(1, 1)
        output, _ = layer.forward(array, BatchConfig(False))
        self.assertEqual(array.shape, output.shape)
        self.assertTrue(np.array_equal(array, output))

    def test_feed_forward(self):
        array = np.random.rand(2, 3, 8, 8)
        for size in range(1, 4):
            for stride in range(1, 4):
                with self.subTest(size=size, stride=stride):
                    expected = manual_pooling(array, size, stride)
                    layer = MaxPoolLayer(size, stride)
                    output, _ = layer.forward(array, BatchConfig(False))
                    self.assertEqual(expected.shape, output.shape)
                    self.assertTrue(np.array_equal(expected, output))

    def test_backward(self):
        batch, channels, height, width = 2, 3, 4, 6
        arr: np.ndarray = np.arange(batch * channels * height * width).reshape((batch, channels, height, width))
        for size in range(2, 4):
            for stride in range(2, 4):
                with self.subTest(size=size, stride=stride):
                    layer = MaxPoolLayer(size, stride)
                    grad = np.random.rand(*get_dims_after_filter(arr.shape, size, stride))
                    result = layer.backward(grad, arr, BatchConfig(False))
                    expected = manual_backprop(arr, grad, size, stride)
                    self.assertGreater(0.01, np.sum(np.abs(result - expected)))


if __name__ == '__main__':
    unittest.main()
