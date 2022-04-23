import unittest

from codebase.persistence import LazyList


class MyTestCase(unittest.TestCase):
    def test_get(self):
        numbers = list(range(10))
        lazy = LazyList(numbers, lambda x: x)
        for i in range(10):
            self.assertEqual(numbers[i], lazy[i])

    def test_conversion(self):
        numbers = list(range(10))
        lazy = LazyList(numbers, lambda x: x * 100)
        for i in range(10):
            self.assertEqual(numbers[i] * 100, lazy[i])

    def test_caching(self):
        def function(x, inc):
            inc[0] += 1
            return x * 100

        transformation_count = [0]
        numbers = list(range(10))
        lazy = LazyList(numbers, lambda x: function(x, transformation_count))
        for i in range(10):
            self.assertEqual(numbers[i] * 100, lazy[i])
        for i in range(10):
            self.assertEqual(numbers[i] * 100, lazy[i])
        for i in range(10):
            self.assertEqual(numbers[i] * 100, lazy[i])

        self.assertEqual(transformation_count[0], 10)

    def test_contains(self):
        numbers = list(range(10))
        lazy = LazyList(numbers, lambda x: x * 100)
        self.assertIn(100, lazy)
        self.assertIn(900, lazy)
        self.assertNotIn(50, lazy)
        self.assertNotIn(99, lazy)
        self.assertNotIn(1500, lazy)

    def test_slicing(self):
        numbers = list(range(10))
        multiplied = list(map(lambda x: x * 100, numbers))
        lazy = LazyList(numbers, lambda x: x * 100)
        self.assertEqual(multiplied[1:3], lazy[1:3])
        self.assertEqual(multiplied[1:5:2], lazy[1:5:2])
        self.assertEqual(multiplied[1:-5:2], lazy[1:-5:2])


if __name__ == '__main__':
    unittest.main()
