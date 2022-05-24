import numpy as np

from codebase.nn.layers import NNLayer
from codebase.nn import BatchConfig


class TanhLayer(NNLayer):
    def forward(self, inputs: np.ndarray, config: BatchConfig) -> tuple[np.ndarray, np.ndarray]:
        result = np.tanh(inputs)
        return result, result

    def backward(self, grad: np.ndarray, cache: np.ndarray, config: BatchConfig):
        return grad * (1 - np.square(cache))
