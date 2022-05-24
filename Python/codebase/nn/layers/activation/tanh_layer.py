import numpy as np

from codebase.nn.layers import NNLayer
from codebase.nn import TrainingConfig


class TanhLayer(NNLayer):
    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        result = np.tanh(inputs)
        return result, result

    def backward(self, grad: np.ndarray, cache: np.ndarray):
        return grad * (1 - np.square(cache))
