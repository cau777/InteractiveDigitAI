from typing import Iterator

import numpy as np

from codebase.nn import TrainingConfig
from codebase.nn.layers import NNLayer
from codebase.nn.lr_optimizers import LrOptimizer

Cache = tuple[np.ndarray, np.ndarray]


class DenseLayer(NNLayer):
    def __init__(self, weights: np.ndarray, biases: np.ndarray, biases_enabled: bool, weights_optimizer: LrOptimizer,
                 biases_optimizer: LrOptimizer, dropout_rate: float = None):
        self.weights = weights
        self.biases = biases
        self.weights_grad = weights * 0
        self.biases_grad = biases * 0
        self.biases_enabled = biases_enabled
        self.weights_optimizer = weights_optimizer
        self.biases_optimizer = biases_optimizer
        self.dropout_rate = dropout_rate

    @staticmethod
    def create_random(in_values: int, out_values: int, weights_optimizer: LrOptimizer, biases_optimizer: LrOptimizer,
                      biases_enabled: bool = True, dropout_rate: float = None):
        if in_values < 1:
            raise ValueError("in_values can't be less than 1")

        if out_values < 1:
            raise ValueError("out_values can't be less than 1")

        std_dev = out_values ** -0.5
        weights = np.random.normal(0, std_dev, (out_values, in_values))
        biases = np.zeros((out_values, 1))
        return DenseLayer(weights, biases, biases_enabled, weights_optimizer, biases_optimizer, dropout_rate)

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, Cache]:
        result = self.weights @ np.expand_dims(inputs, -1)
        if self.biases_enabled:
            result += self.biases

        dropout = None
        if self.dropout_rate is not None:
            dropout = np.random.rand(*result.shape) > self.dropout_rate
            result *= dropout

        return np.squeeze(result, -1), (inputs, dropout)

    def backward(self, grad: np.ndarray, cache: Cache) -> np.ndarray:
        inputs, dropout = cache

        grad = np.expand_dims(grad, -1)
        if dropout is not None:
            grad *= dropout

        inputs = np.expand_dims(inputs, -2)

        weights_error = grad @ inputs
        self.weights_grad += weights_error.mean(0)

        if self.biases_enabled:
            self.biases_grad += grad.mean(0)

        return np.squeeze(self.weights.T @ grad, -1)

    def train(self, config: TrainingConfig):
        self.weights += self.weights_optimizer.optimize(self.weights_grad, config)
        self.weights_grad *= 0

        if self.biases_enabled:
            self.biases += self.biases_optimizer.optimize(self.biases_grad, config)
            self.biases_grad *= 0

    def trainable_params_count(self) -> int:
        return self.weights.size + self.biases.size

    def get_trainable_params(self) -> list[float]:
        return list(self.weights.flat) + list(self.biases.flat)

    def set_trainable_params(self, params_iterator: Iterator[float]) -> None:
        self.weights = np.array([next(params_iterator) for _ in range(self.weights.size)]).reshape(self.weights.shape)
        self.biases = np.array([next(params_iterator) for _ in range(self.biases.size)]).reshape(self.biases.shape)
