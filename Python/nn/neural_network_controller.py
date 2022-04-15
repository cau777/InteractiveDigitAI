import numpy as np

from nn import TrainingConfig, TrainingExample
from nn.layers import NNLayer
from nn.loss_functions import LossFunction
from nn.utils import select_random
from nn.measure_trackers import create_tracker


class NeuralNetworkController:
    def __init__(self, main_layer: NNLayer, loss_func: LossFunction, version: int = 1):
        self.main_layer = main_layer
        self.loss_func = loss_func
        self.version = version

    def evaluate(self, inputs: np.ndarray):
        return self.main_layer.feed_forward(inputs)

    def classify(self, inputs: np.ndarray):
        return self.main_layer.feed_forward(inputs).argmax()

    def train(self, data: list[TrainingExample], epochs: int, batch_size: int = 16, measure: list[str] = None) \
            -> list[dict[str]]:

        measures_trackers = [create_tracker(m) for m in measure]
        measures_result = []

        # TODO: mini-batch arrays
        for e in range(epochs):
            config = TrainingConfig(self.version, batch_size)
            mini_batch: list[TrainingExample] = select_random(data, batch_size)

            for example in mini_batch:
                inputs, label = example.inputs, example.label

                outputs = self.main_layer.feed_forward(inputs)
                loss = self.loss_func.calc_loss(label, outputs)
                loss_grad = self.loss_func.calc_loss_gradient(label, outputs)

                self.main_layer.backpropagate_gradient(inputs, outputs, loss_grad, config)
                for t in measures_trackers:
                    t.track(inputs, outputs, label, loss)

            batch_measures = dict()
            for t in measures_trackers:
                t.record(batch_measures)
            measures_result.append(batch_measures)

            self.main_layer.train(config)
            self.version += 1
            print(self.version, batch_measures)

        return measures_result

    def test(self, data: list[TrainingExample]):
        total_loss = 0

        for example in data:
            outputs = self.main_layer.feed_forward(example.inputs)
            loss = self.loss_func.calc_loss(example.label, outputs)
            total_loss += loss.sum() / loss.size

        return total_loss / len(data)
