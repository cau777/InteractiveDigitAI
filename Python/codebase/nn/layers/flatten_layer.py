import numpy as np

from codebase.nn.layers import NNLayer


class FlattenLayer(NNLayer):
    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        return inputs.reshape(inputs.shape[0], -1), inputs.shape

    def backward(self, grad: np.ndarray, cache: tuple[int, ...]):
        return grad.reshape(cache)
