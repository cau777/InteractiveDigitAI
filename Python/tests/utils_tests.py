import unittest

from libs.nn.utils import get_dims_after_filter, split_array


class MyTestCase(unittest.TestCase):
    def test_get_dims_after_filter(self):
        self.assertEqual((3, 3), get_dims_after_filter((3, 3), 1, 1))
        self.assertEqual((1, 4, 4), get_dims_after_filter((1, 4, 4), 1, 1))
        self.assertEqual((1, 1, 4, 4), get_dims_after_filter((1, 1, 4, 4), 1, 1))

        self.assertEqual((1, 1, 2, 2), get_dims_after_filter((1, 1, 4, 4), 3, 1))
        self.assertEqual((1, 1, 3, 3), get_dims_after_filter((1, 1, 4, 4), 2, 1))
        self.assertEqual((1, 1, 3, 3), get_dims_after_filter((1, 1, 4, 4), 2, 1))

        self.assertEqual((1, 1, 2, 2), get_dims_after_filter((1, 1, 6, 6), 2, 3))

    def test_split_array(self):
        self.assertEqual([[1, 2, 3], [4, 5, 6]], split_array([1, 2, 3, 4, 5, 6], 3))
        self.assertEqual([[1, 2, 3], [4, 5, 6], [7]], split_array([1, 2, 3, 4, 5, 6, 7], 3))



if __name__ == '__main__':
    unittest.main()
