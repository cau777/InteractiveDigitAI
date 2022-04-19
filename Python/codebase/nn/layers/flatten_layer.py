import numpy as np

from codebase.nn.layers import NNLayer
from codebase.nn import TrainingConfig


class FlattenLayer(NNLayer):
    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        return inputs.reshape(inputs.shape[0], -1)

    def backpropagate_gradient(self, inputs: np.ndarray, outputs: np.ndarray, current_gradient: np.ndarray,
                               config: TrainingConfig):
        return current_gradient.reshape(inputs.shape)
