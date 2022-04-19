import numpy as np

from codebase.nn import TrainingConfig, TrainingExample
from codebase.nn.layers import NNLayer
from codebase.nn.loss_functions import LossFunction
from codebase.nn.utils import select_random, split_array
from codebase.nn.measure_trackers import create_tracker


class NeuralNetworkController:
    def __init__(self, main_layer: NNLayer, loss_func: LossFunction, version: int = 1):
        self.main_layer = main_layer
        self.loss_func = loss_func
        self.version = version

    def evaluate_single(self, inputs: np.ndarray) -> np.ndarray:
        return self.evaluate_batch(np.stack([inputs]))[0]

    def evaluate_batch(self, inputs: np.ndarray):
        return self.main_layer.feed_forward(inputs)

    def classify_single(self, inputs: np.ndarray) -> int:
        return self.classify_batch(np.stack([inputs]))[0]

    def classify_batch(self, inputs: np.ndarray):
        print(self.main_layer.feed_forward(inputs).shape)
        return np.argmax(self.main_layer.feed_forward(inputs), -1)

    def train(self, data: list[TrainingExample], epochs: int, batch_size: int = 16, measure: list[str] = None) \
            -> list[dict[str]]:
        if not measure:
            measure = ["avg_loss"]

        measures_trackers = [create_tracker(m) for m in measure]
        measures_result = []

        for e in range(epochs):
            config = TrainingConfig(self.version, batch_size)

            mini_batch_examples: list[TrainingExample] = select_random(data, batch_size)
            inputs = np.stack([example.inputs for example in mini_batch_examples])
            labels = np.stack([example.label for example in mini_batch_examples])

            outputs = self.main_layer.feed_forward(inputs)
            loss = self.loss_func.calc_loss(labels, outputs)
            loss_grad = self.loss_func.calc_loss_gradient(labels, outputs)

            self.main_layer.backpropagate_gradient(inputs, outputs, loss_grad, config)

            for b in range(batch_size):
                for t in measures_trackers:
                    t.track(inputs[b], outputs[b], labels[b], loss[b])

            batch_measures = dict()
            for t in measures_trackers:
                t.record(batch_measures)
            measures_result.append(batch_measures)

            self.main_layer.train(config)
            self.version += 1
            print(self.version, batch_measures)

        return measures_result

    def test(self, data: list[TrainingExample], measure: list[str] = None, batch_size: int = 16):
        if not measure:
            measure = ["avg_loss"]

        measures_trackers = [create_tracker(m) for m in measure]

        for dataset in split_array(data, batch_size):
            inputs = np.stack([example.inputs for example in dataset])
            labels = np.stack([example.label for example in dataset])
            outputs = self.main_layer.feed_forward(inputs)
            loss = self.loss_func.calc_loss(labels, outputs)

            for b in range(len(dataset)):
                for t in measures_trackers:
                    t.track(inputs[b], outputs[b], labels[b], loss[b])

        measures_result = dict()
        for t in measures_trackers:
            t.record(measures_result)
        return measures_result
