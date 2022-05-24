import unittest

from codebase.general_utils import split_array


class MyTestCase(unittest.TestCase):
    def test_split_array(self):
        self.assertEqual([[1, 2, 3], [4, 5, 6]], split_array([1, 2, 3, 4, 5, 6], 3))
        self.assertEqual([[1, 2, 3], [4, 5, 6], [7]], split_array([1, 2, 3, 4, 5, 6, 7], 3))


if __name__ == '__main__':
    unittest.main()
