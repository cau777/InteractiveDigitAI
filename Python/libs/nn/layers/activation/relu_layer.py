import numpy as np

from libs.nn.layers import NNLayer
from libs.nn import TrainingConfig


def relu(x):
    return max(0, x)


def relu_gradient(x):
    return 1 if x > 0 else 0


v_relu = np.vectorize(relu)
v_relu_gradient = np.vectorize(relu_gradient)


class ReluLayer(NNLayer):
    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        return v_relu(inputs)

    def backpropagate_gradient(self, inputs: np.ndarray, outputs: np.ndarray, current_gradient: np.ndarray,
                               config: TrainingConfig):
        return current_gradient * v_relu_gradient(inputs)
