import random
import unittest

import numpy as np

from codebase.nn import NeuralNetworkController, TrainingExample
from codebase.nn.layers import SequentialLayer, DenseLayer, FlattenLayer, ConvolutionLayer, ReshapeLayer, MaxPoolLayer
from codebase.nn.layers.activation import TanhLayer, ReluLayer
from codebase.nn.loss_functions import MseLossFunction, CrossEntropyLossFunction
from codebase.nn.lr_optimizers import AdamLrOptimizer
from codebase.nn.utils import init_random


def conv_controller():
    return NeuralNetworkController(SequentialLayer(
        ReshapeLayer((1, 28, 28)),

        ConvolutionLayer.create_random(1, 32, 3, AdamLrOptimizer(0.01)),
        ReluLayer(),
        MaxPoolLayer(2, 2),

        FlattenLayer(),

        DenseLayer.create_random(5408, 100, AdamLrOptimizer(), AdamLrOptimizer()),
        ReluLayer(),
        DenseLayer.create_random(100, 10, AdamLrOptimizer(), AdamLrOptimizer())
    ), CrossEntropyLossFunction())


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        init_random()

    def test_eval_empty(self):
        controller = NeuralNetworkController(SequentialLayer(), MseLossFunction())
        inputs = np.random.rand(10, 10).astype("float32")
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

    @unittest.skip
    def test_classification_conv(self):
        controller = conv_controller()

        examples = [TrainingExample(np.random.rand(28, 28), np.zeros(10)) for _ in range(64)]
        for e in examples:
            e.label[random.randint(0, 9)] = 1.0
        controller.train(examples, 50, measure=["accuracy"])
        results = controller.test(examples, measure=["accuracy"])

        self.assertGreater(results["accuracy"], 0.8)
        # self.assertLess(0.5, results["accuracy"])

    @unittest.skip
    def test_benchmark(self):
        controller = conv_controller()
        print(controller.benchmark((28, 28)))


if __name__ == '__main__':
    unittest.main()
