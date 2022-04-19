import numpy as np

from codebase.nn.layers import NNLayer
from codebase.nn import TrainingConfig
from codebase.nn.utils import get_dims_after_filter


class MaxPoolLayer(NNLayer):
    def __init__(self, size: int, stride: int):
        self.size = size
        self.stride = stride

    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        batch_size, channels, new_height, new_width = get_dims_after_filter(inputs.shape, self.size, self.stride)
        result = np.zeros((batch_size, channels, new_height, new_width))

        for b in range(batch_size):
            for c in range(channels):
                for h in range(new_height):
                    for w in range(new_width):
                        h_offset = h * self.stride
                        w_offset = w * self.stride
                        area: np.ndarray = inputs[b, c, h_offset:h_offset + self.size, w_offset:w_offset + self.size]
                        result[b, c, h, w] = np.max(area)  # TODO Improve

        return result

    def backpropagate_gradient(self, inputs: np.ndarray, outputs: np.ndarray, current_gradient: np.ndarray,
                               config: TrainingConfig):
        batch_size, channels, new_height, new_width = get_dims_after_filter(inputs.shape, self.size, self.stride)

        result = inputs * 0
        for b in range(batch_size):
            for c in range(channels):
                for h in range(new_height):
                    for w in range(new_width):
                        h_offset = h * self.stride
                        w_offset = w * self.stride
                        area: np.ndarray = inputs[b, c, h_offset:h_offset + self.size, w_offset:w_offset + self.size]
                        index = np.unravel_index(np.argmax(area), area.shape)
                        result[(*index,)] += current_gradient[b, c, h, w]
        return result
