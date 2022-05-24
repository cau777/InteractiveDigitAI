import numpy as np

from codebase.nn import BatchConfig
from codebase.nn.layers import NNLayer


class DropoutLayer(NNLayer):
    def __init__(self, rate: float):
        self.rate = rate

    def forward(self, inputs: np.ndarray, config: BatchConfig) -> tuple[np.ndarray, np.ndarray]:
        dropout = None
        if config.training:
            dropout = np.random.rand(*inputs.shape) > self.rate
            inputs *= dropout
        return inputs, dropout

    def backward(self, grad: np.ndarray, cache: np.ndarray, config: BatchConfig) -> np.ndarray:
        return grad * cache
