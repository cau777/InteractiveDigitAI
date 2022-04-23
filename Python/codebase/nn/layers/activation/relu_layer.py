import numpy as np

from codebase.nn.layers import NNLayer
from codebase.nn import TrainingConfig


class ReluLayer(NNLayer):
    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        return inputs * (inputs > 0)

    def backpropagate_gradient(self, inputs: np.ndarray, outputs: np.ndarray, current_gradient: np.ndarray,
                               config: TrainingConfig):
        return current_gradient * (inputs > 0)
