from typing import Iterator

import numpy as np

from libs.nn.layers import NNLayer
from libs.nn.lr_optimizers import LrOptimizer
from libs.nn import TrainingConfig


class DenseLayer(NNLayer):
    def __init__(self, weights: np.ndarray, biases: np.ndarray, biases_enabled: bool, weights_optimizer: LrOptimizer,
                 biases_optimizer: LrOptimizer):
        self.weights = weights
        self.biases = biases
        self.weights_grad = weights * 0
        self.biases_grad = biases * 0
        self.biases_enabled = biases_enabled
        self.weights_optimizer = weights_optimizer
        self.biases_optimizer = biases_optimizer

    @staticmethod
    def create_random(in_values: int, out_values: int, weights_optimizer: LrOptimizer, biases_optimizer: LrOptimizer,
                      biases_enabled: bool = True):
        if in_values < 1:
            raise ValueError("in_values can't be less than 1")

        if out_values < 1:
            raise ValueError("out_values can't be less than 1")

        std_dev = out_values ** -0.5
        weights = np.random.normal(0, std_dev, (out_values, in_values))
        biases = np.zeros((out_values, 1))
        return DenseLayer(weights, biases, biases_enabled, weights_optimizer, biases_optimizer)

    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        result = self.weights @ np.expand_dims(inputs, -1)
        if self.biases_enabled:
            result += self.biases
        return np.squeeze(result, -1)

    def backpropagate_gradient(self, inputs: np.ndarray, outputs: np.ndarray, current_gradient: np.ndarray,
                               config: TrainingConfig):
        factor = 1
        current_gradient = np.expand_dims(current_gradient, -1)
        inputs = np.expand_dims(inputs, -2)

        weights_error = current_gradient @ inputs
        self.weights_grad += (factor * weights_error).sum(0)

        if self.biases_enabled:
            self.biases_grad += (factor * current_gradient).sum(0)

        return np.squeeze(self.weights.T @ current_gradient, -1)

    def train(self, config: TrainingConfig):
        self.weights += self.weights_optimizer.optimize(self.weights_grad, config)
        self.weights_grad *= 0

        if self.biases_enabled:
            self.biases += self.biases_optimizer.optimize(self.biases_grad, config)
            self.biases_grad *= 0

    def trainable_params_count(self) -> int:
        return self.weights.size + self.biases.size

    def trainable_params(self) -> list[float]:
        return list(self.weights.flat) + list(self.biases.flat)

    def set_trainable_params(self, params_iterator: Iterator[float]) -> None:
        self.weights = np.array([next(params_iterator) for _ in range(self.weights.size)]).reshape(self.weights.shape)
        self.biases = np.array([next(params_iterator) for _ in range(self.biases.size)]).reshape(self.biases.shape)
