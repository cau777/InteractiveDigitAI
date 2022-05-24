import unittest

from codebase.nn.utils import get_dims_after_filter


class MyTestCase(unittest.TestCase):
    def test_get_dims_after_filter(self):
        self.assertEqual((3, 3), get_dims_after_filter((3, 3), 1, 1))
        self.assertEqual((1, 4, 4), get_dims_after_filter((1, 4, 4), 1, 1))
        self.assertEqual((1, 1, 4, 4), get_dims_after_filter((1, 1, 4, 4), 1, 1))

        self.assertEqual((1, 1, 2, 2), get_dims_after_filter((1, 1, 4, 4), 3, 1))
        self.assertEqual((1, 1, 3, 3), get_dims_after_filter((1, 1, 4, 4), 2, 1))
        self.assertEqual((1, 1, 3, 3), get_dims_after_filter((1, 1, 4, 4), 2, 1))

        self.assertEqual((1, 1, 2, 2), get_dims_after_filter((1, 1, 6, 6), 2, 3))


if __name__ == '__main__':
    unittest.main()
