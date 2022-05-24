import numpy as np

from codebase.nn.layers import NNLayer


class SigmoidLayer(NNLayer):
    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        result = 1 / (1 + np.exp(-inputs))
        return result, result

    def backward(self, grad: np.ndarray, cache: np.ndarray):
        return grad * cache * (1 - cache)
