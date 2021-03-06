import numpy as np

from codebase.nn import BatchConfig
from codebase.nn.layers import NNLayer


class ReshapeLayer(NNLayer):
    def __init__(self, shape: tuple[int, ...]):
        if -1 in shape:
            raise ValueError("Shape can't contain -1")
        self.shape = shape

    def forward(self, inputs: np.ndarray, config: BatchConfig) -> tuple[np.ndarray, tuple[int, ...]]:
        return inputs.reshape((inputs.shape[0], *self.shape)), inputs.shape

    def backward(self, grad: np.ndarray, cache: tuple[int, ...], config: BatchConfig):
        return grad.reshape(cache)
