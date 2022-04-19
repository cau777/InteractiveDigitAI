import unittest

import numpy as np

from codebase.nn import NeuralNetworkController, TrainingExample
from codebase.nn.layers import SequentialLayer, DenseLayer
from codebase.nn.layers.activation import TanhLayer
from codebase.nn.loss_functions import MseLossFunction
from codebase.nn.lr_optimizers import AdamLrOptimizer
from codebase.nn.utils import init_random


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        init_random()

    def test_eval_empty(self):
        controller = NeuralNetworkController(SequentialLayer(), MseLossFunction())
        inputs = np.random.rand(10, 10)
        result = controller.evaluate_single(inputs)
        self.assertTrue(np.array_equal(inputs, result))

    def test_eval_dense(self):
        controller = NeuralNetworkController(SequentialLayer(
            DenseLayer.create_random(10, 64, AdamLrOptimizer(), AdamLrOptimizer()),
            TanhLayer(),
            DenseLayer.create_random(64, 64, AdamLrOptimizer(), AdamLrOptimizer()),
            TanhLayer(),
            DenseLayer.create_random(64, 10, AdamLrOptimizer(), AdamLrOptimizer()),
        ), MseLossFunction())

        examples = [TrainingExample(np.random.rand(10), np.random.rand(10)) for _ in range(64)]

        controller.train(examples, 100, measure=["avg_loss"])
        results = controller.test(examples, measure=["avg_loss"])
        self.assertGreater(0.1, results["avg_loss"])


if __name__ == '__main__':
    unittest.main()
