import numpy as np
from typing import Iterator
from codebase.nn.layers import NNLayer
from codebase.nn import TrainingConfig


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
                               config: TrainingConfig) -> np.ndarray:
        if self.layers_inputs is None:
            raise ValueError("feed_forward should be called before backpropagate_gradient")

        gradient = current_gradient
        for i in range(len(self.layers) - 1, -1, -1):
            gradient = self.layers[i].backpropagate_gradient(self.layers_inputs[i], self.layers_inputs[i + 1], gradient,
                                                             config)

        self.layers_inputs = None
        return gradient

    def train(self, config: TrainingConfig):
        for layer in self.layers:
            layer.train(config)

    def trainable_params_count(self) -> int:
        return sum(map(lambda layer: layer.trainable_params_count, self.layers))

    def get_trainable_params(self) -> list[float]:
        result = []
        for layer in self.layers:
            result.extend(layer.get_trainable_params())
        return result

    def set_trainable_params(self, params_iterator: Iterator[float]) -> None:
        for layer in self.layers:
            layer.set_trainable_params(params_iterator)

    def benchmark_feed_forward(self, inputs: np.ndarray, results: list[tuple[str, float]]) -> np.ndarray:
        self.layers_inputs = list([None] * (len(self.layers) + 1))
        self.layers_inputs[0] = inputs

        for i in range(len(self.layers)):
            self.layers_inputs[i + 1] = self.layers[i].benchmark_feed_forward(self.layers_inputs[i], results)

        return self.layers_inputs[-1]

    def benchmark_backprapagate(self, inputs: np.ndarray, outputs: np.ndarray, current_gradient: np.ndarray,
                                config: TrainingConfig, results: list[tuple[str, float]]) -> np.ndarray:
        gradient = current_gradient
        for i in range(len(self.layers) - 1, -1, -1):
            gradient = self.layers[i].benchmark_backprapagate(self.layers_inputs[i], self.layers_inputs[i + 1],
                                                              gradient, config, results)
        return gradient

    def benchmark_train(self, config: TrainingConfig, results: list[tuple[str, float]]):
        for layer in self.layers:
            layer.benchmark_train(config, results)
