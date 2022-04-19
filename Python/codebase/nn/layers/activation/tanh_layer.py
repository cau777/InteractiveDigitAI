import numpy as np

from codebase.nn.layers import NNLayer
from codebase.nn import TrainingConfig


class TanhLayer(NNLayer):
    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.tanh(inputs)

    def backpropagate_gradient(self, inputs: np.ndarray, outputs: np.ndarray, current_gradient: np.ndarray,
                               config: TrainingConfig):
        return current_gradient * (1 - np.square(outputs))
