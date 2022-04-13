import numpy as np

from nn.layers.nn_layer import NNLayer
from nn.training_config import TrainingConfig
from nn.utils import get_dims_after_filter


class MaxPoolLayer(NNLayer):
    def __init__(self, size: int, stride: int):
        self.size = size
        self.stride = stride

    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        channels, new_height, new_width = get_dims_after_filter(inputs.shape, self.size, self.stride)
        result = np.zeros((channels, new_height, new_width))

        for c in range(channels):
            for h in range(new_height):
                for w in range(new_width):
                    h_offset = h * self.stride
                    w_offset = w * self.stride
                    area: np.ndarray = inputs[c, h_offset:h_offset + self.size, w_offset:w_offset + self.size]
                    result[c, h, w] = np.max(area)  # TODO Improve

        return result

    def backpropagate_gradient(self, inputs: np.ndarray, outputs: np.ndarray, current_gradient: np.ndarray,
                               config: TrainingConfig):
        channels, new_height, new_width = get_dims_after_filter(inputs.shape, self.size, self.stride)

        result = inputs * 0
        for c in range(channels):
            for h in range(new_height):
                for w in range(new_width):
                    h_offset = h * self.stride
                    w_offset = w * self.stride
                    area: np.ndarray = inputs[c, h_offset:h_offset + self.size, w_offset:w_offset + self.size]
                    index = np.unravel_index(np.argmax(area), area.shape)
                    result[(*index,)] += current_gradient[c, h, w]
        return result
