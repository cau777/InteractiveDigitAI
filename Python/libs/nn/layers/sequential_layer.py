from typing import Iterator

import numpy as np

from libs.nn.layers import NNLayer
from libs.nn import TrainingConfig


class SequentialLayer(NNLayer):
    def __init__(self, *layers: NNLayer):
        self.layers = list(layers)
        self.layers_inputs: list[np.ndarray] | None = None

    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        self.layers_inputs = list([None] * (len(self.layers) + 1))
        self.layers_inputs[0] = inputs

        for i in range(len(self.layers)):
            self.layers_inputs[i + 1] = self.layers[i].feed_forward(self.layers_inputs[i])

        return self.layers_inputs[-1]

    def backpropagate_gradient(self, inputs: np.ndarray, outputs: np.ndarray, current_gradient: np.ndarray,
                               config: TrainingConfig):
        if self.layers_inputs is None:
            raise ValueError("feed_forward should be called before backpropagate_gradient")

        gradient = current_gradient
        for i in range(len(self.layers) - 1, -1, -1):
            gradient = self.layers[i].backpropagate_gradient(self.layers_inputs[i], self.layers_inputs[i + 1], gradient,
                                                             config)

        self.layers_inputs = None

    def train(self, config: TrainingConfig):
        for layer in self.layers:
            layer.train(config)

    def trainable_params_count(self) -> int:
        return sum(map(lambda layer: layer.trainable_params_count, self.layers))

    def trainable_params(self) -> list[float]:
        result = []
        for layer in self.layers:
            result.extend(layer.trainable_params)
        return result

    def set_trainable_params(self, params_iterator: Iterator[float]) -> None:
        for layer in self.layers:
            layer.set_trainable_params(params_iterator)
