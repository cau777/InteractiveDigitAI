import numpy as np

from nn.layers.nn_layer import NNLayer
from nn.training_config import TrainingConfig


class SigmoidLayer(NNLayer):
    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-inputs))

    def backpropagate_gradient(self, inputs: np.ndarray, outputs: np.ndarray, current_gradient: np.ndarray,
                               config: TrainingConfig):
        return current_gradient * outputs * (1 - outputs)
