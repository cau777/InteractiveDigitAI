import numpy as np

from codebase.nn.layers import NNLayer
from codebase.nn import TrainingConfig


class SigmoidLayer(NNLayer):
    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-inputs))

    def backpropagate_gradient(self, inputs: np.ndarray, outputs: np.ndarray, current_gradient: np.ndarray,
                               config: TrainingConfig):
        return current_gradient * outputs * (1 - outputs)
