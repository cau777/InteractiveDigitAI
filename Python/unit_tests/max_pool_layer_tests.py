import unittest

import numpy as np

from nn.utils import init_random
from nn.layers.max_pool import MaxPoolLayer
from nn.utils import get_dims_after_filter


def manual_pooling(array: np.ndarray, size: int, stride: int):
    batch_size, channels, new_height, new_width = get_dims_after_filter(array.shape, size, stride)
    result = np.zeros((batch_size, channels, new_height, new_width))

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


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        init_random()

    def test_feed_forward0(self):
        array = np.random.rand(1, 1, 8, 8)
        layer = MaxPoolLayer(1, 1)
        output = layer.feed_forward(array)
        self.assertEqual(array.shape, output.shape)
        self.assertTrue(np.array_equal(array, output))

    def test_feed_forward(self):
        array = np.random.rand(2, 3, 8, 8)
        for size in range(1, 4):
            for stride in range(1, 4):
                with self.subTest(size=size, stride=stride):
                    expected = manual_pooling(array, size, stride)
                    layer = MaxPoolLayer(size, stride)
                    output = layer.feed_forward(array)
                    self.assertEqual(expected.shape, output.shape)
                    self.assertTrue(np.array_equal(expected, output))


if __name__ == '__main__':
    unittest.main()
