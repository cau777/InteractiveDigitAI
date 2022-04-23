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

        for h in range(new_height):
            for w in range(new_width):
                h_offset = h * self.stride
                w_offset = w * self.stride
                area: np.ndarray = inputs[:, :, h_offset:h_offset + self.size, w_offset:w_offset + self.size]
                reshaped = area.reshape((batch_size, channels, -1))
                result[:, :, h, w] = np.max(reshaped, -1)

        return result

    def backpropagate_gradient(self, inputs: np.ndarray, outputs: np.ndarray, current_gradient: np.ndarray,
                               config: TrainingConfig):
        batch_size, channels, new_height, new_width = get_dims_after_filter(inputs.shape, self.size, self.stride)
        result: np.ndarray = inputs * 0.0

        for h in range(new_height):
            for w in range(new_width):
                h_offset = h * self.stride
                w_offset = w * self.stride
                area: np.ndarray = inputs[:, :, h_offset:h_offset + self.size, w_offset:w_offset + self.size]
                reshaped = area.reshape((batch_size, channels, -1))

                indices = np.argmax(reshaped, -1)
                unraveled_h, unraveled_w = np.unravel_index(indices, area.shape[-2:])
                unraveled_h += h_offset
                unraveled_w += w_offset
                for b in range(batch_size):
                    for c in range(channels):
                        result[b, c, unraveled_h[b][c], unraveled_w[b][c]] += current_gradient[b, c, h, w]

        return result
