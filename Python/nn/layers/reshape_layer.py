import numpy as np

from nn.layers.nn_layer import NNLayer
from nn.training_config import TrainingConfig


class ReshapeLayer(NNLayer):
    def __init__(self, shape: tuple[int]):
        if -1 in shape:
            raise ValueError("Shape can't contain -1")
        self.shape = shape

    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        return inputs.reshape(self.shape)

    def backpropagate_gradient(self, inputs: np.ndarray, outputs: np.ndarray, current_gradient: np.ndarray,
                               config: TrainingConfig):
        return current_gradient.reshape(inputs.shape)



