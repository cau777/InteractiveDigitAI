import math
from typing import Sequence

import numpy as np

from codebase.general_utils import split_array
from codebase.nn import TrainingConfig, TrainingExample
from codebase.nn.layers import NNLayer
from codebase.nn.loss_functions import LossFunction
from codebase.nn.utils import select_random
from codebase.nn.measure_trackers import create_tracker


def find_best_batch_size():
    return 64


# TODO: find best batch size
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
        return np.argmax(self.main_layer.feed_forward(inputs), -1)

    def train(self, data: Sequence[TrainingExample], epochs: int, batch_size: int = -1, measure: list[str] = None) \
            -> list[dict[str]]:
        if batch_size == -1:
            batch_size = find_best_batch_size()

        if not measure:
            measure = ["avg_loss"]

        measures_trackers = [create_tracker(m) for m in measure]
        measures_result = []

        print(f"Started training {epochs} epochs. Batch size={batch_size}")

        for e in range(epochs):
            print(f"Started {e} epoch")
            config = TrainingConfig(self.version, batch_size)

            mini_batch_examples: list[TrainingExample] = select_random(data, batch_size)
            inputs = np.stack([example.inputs for example in mini_batch_examples])
            labels = np.stack([example.label for example in mini_batch_examples])

            # print("Forward propagating inputs")
            outputs = self.main_layer.feed_forward(inputs)
            loss = self.loss_func.calc_loss(labels, outputs)
            loss_grad = self.loss_func.calc_loss_gradient(labels, outputs)

            # print("Back propagating gradients")
            self.main_layer.backpropagate_gradient(inputs, outputs, loss_grad, config)

            for b in range(batch_size):
                for t in measures_trackers:
                    t.track(inputs[b], outputs[b], labels[b], loss[b])

            batch_measures = dict()
            for t in measures_trackers:
                t.record(batch_measures)
            measures_result.append(batch_measures)

            # print("Training")
            self.main_layer.train(config)
            self.version += 1
            print(f"Finished epoch {e} with {batch_measures}. Version={self.version}")

        return measures_result

    def test(self, data: Sequence[TrainingExample], measure: list[str] = None, batch_size: int = -1):
        if batch_size == -1:
            batch_size = find_best_batch_size()

        if not measure:
            measure = ["avg_loss"]

        measures_trackers = [create_tracker(m) for m in measure]
        split = split_array(data, batch_size)
        print(f"Started testing. Data split in {len(split)} batches of 16")

        for index, dataset in enumerate(split):
            print(f"Started test batch {index}")

            inputs = np.stack([example.inputs for example in dataset])
            labels = np.stack([example.label for example in dataset])
            outputs = self.main_layer.feed_forward(inputs)
            loss = self.loss_func.calc_loss(labels, outputs)

            for b in range(len(dataset)):
                for t in measures_trackers:
                    t.track(inputs[b], outputs[b], labels[b], loss[b])
            print(f"Finished testing batch {index}")

        measures_result = dict()
        for t in measures_trackers:
            t.record(measures_result)
        return measures_result

    def benchmark(self, inputs_shape: tuple[int, ...], batch: int = 16):
        inputs = np.abs(np.random.rand(batch, *inputs_shape))
        config = TrainingConfig(10, batch)
        forward_times = []
        backward_times = []
        train_times = []

        outputs = self.main_layer.benchmark_feed_forward(inputs, forward_times)
        self.main_layer.benchmark_backprapagate(inputs, outputs, outputs / 2, config, backward_times)
        self.main_layer.benchmark_train(config, train_times)
        return {
            "forward": forward_times,
            "backward": backward_times,
            "train": train_times
        }
