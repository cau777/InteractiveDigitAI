import numpy as np

from nn.layers import NNLayer
from nn import TrainingConfig


class FlattenLayer(NNLayer):
    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        return inputs.reshape(-1, 1)

    def backpropagate_gradient(self, inputs: np.ndarray, outputs: np.ndarray, current_gradient: np.ndarray,
                               config: TrainingConfig):
        return current_gradient.reshape(inputs.shape)
