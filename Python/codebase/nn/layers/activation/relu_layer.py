import numpy as np

from codebase.nn import BatchConfig
from codebase.nn.layers import NNLayer


class ReluLayer(NNLayer):
    def forward(self, inputs: np.ndarray, config: BatchConfig) -> tuple[np.ndarray, np.ndarray]:
        positives = inputs > 0
        return inputs * positives, positives

    def backward(self, grad: np.ndarray, cache: np.ndarray, config: BatchConfig):
        return grad * cache
