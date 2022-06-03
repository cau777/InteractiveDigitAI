import numpy as np
from typing import Iterator
from codebase.nn.layers import NNLayer
from codebase.nn import BatchConfig


class SequentialLayer(NNLayer):
    def __init__(self, *layers: NNLayer):
        self.layers = list(layers)

    def forward(self, inputs: np.ndarray, config: BatchConfig) -> tuple[np.ndarray, list]:
        cache: list = [None] * len(self.layers)
        layer_inputs = inputs

        for i in range(len(self.layers)):
            layer_inputs, cache[i] = self.layers[i].forward(layer_inputs, config)

        return layer_inputs, cache

    def backward(self, grad: np.ndarray, cache: list, config: BatchConfig) -> np.ndarray:
        gradient = grad
        for i in range(len(self.layers) - 1, -1, -1):
            gradient = self.layers[i].backward(gradient, cache[i], config)

        return gradient

    def train(self, config: BatchConfig):
        for layer in self.layers:
            layer.train(config)

    def count_trainable_params(self) -> int:
        return sum(map(lambda layer: layer.count_trainable_params(), self.layers))

    def get_trainable_params(self) -> list[float]:
        result = []
        for layer in self.layers:
            result.extend(layer.get_trainable_params())
        return result

    def set_trainable_params(self, params_iterator: Iterator[float]) -> None:
        for layer in self.layers:
            layer.set_trainable_params(params_iterator)

    def benchmark_forward(self, inputs: np.ndarray, results: list[tuple[str, float]], config: BatchConfig) \
            -> tuple[np.ndarray, object]:
        cache: list = [None] * len(self.layers)
        layer_inputs = inputs

        for i in range(len(self.layers)):
            layer_inputs, cache[i] = self.layers[i].benchmark_forward(layer_inputs, results, config)

        return layer_inputs, cache

    def benchmark_backward(self, current_gradient: np.ndarray, cache: list, results: list[tuple[str, float]],
                           config: BatchConfig) -> np.ndarray:
        gradient = current_gradient
        for i in range(len(self.layers) - 1, -1, -1):
            gradient = self.layers[i].benchmark_backward(gradient, cache[i], results, config)
        return gradient

    def benchmark_train(self, config: BatchConfig, results: list[tuple[str, float]]):
        for layer in self.layers:
            layer.benchmark_train(config, results)
