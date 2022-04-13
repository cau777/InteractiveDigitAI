import numpy as np

from nn.layers.nn_layer import NNLayer
from nn.lr_optimizers.lr_optimizer import LrOptimizer
from nn.training_config import TrainingConfig
from nn.utils import get_dims_after_filter


def build_random(out_channels: int, in_channels: int, kernel_size: int, optimizer: LrOptimizer, stride: int = 1,
                 padding: int = 0):
    if stride < 1:
        raise ValueError("Stride can't be less than 1")
    if padding < 0:
        raise ValueError("Padding can't be negative")

    std_dev = (out_channels * kernel_size * kernel_size) ** -0.5  # TODO: improve initialization
    kernels = np.random.normal(0, std_dev, (out_channels, in_channels, kernel_size, kernel_size))
    return ConvolutionLayer(kernels, optimizer, stride, padding)


def pad3d(array: np.ndarray, padding: int):
    height = array.shape[1]
    width = array.shape[2]
    result = np.zeros((array.shape[0], height + 2 * padding, width + 2 * padding))
    result[:, padding:height + padding, padding:width + padding] = array
    return result


def remove_padding3d(array: np.ndarray, padding: int):
    shape = array.shape
    return array[:, padding:shape[1] - padding, padding: shape[2] - padding]


def extract_fragments3d(array: np.ndarray, size: int, stride: int):
    channels = array.shape[0]
    new_height = (array.shape[1] - size) // stride + 1
    new_width = (array.shape[2] - size) // stride + 1

    result = np.zeros((new_height, new_width, channels, size, size))
    for h in range(new_height):
        for w in range(new_width):
            h_offset = h * stride
            w_offset = w * stride
            result[h][w] = array[:, h_offset:h_offset + size, w_offset:w_offset + size]
    return result


class ConvolutionLayer(NNLayer):
    def __init__(self, kernels: np.ndarray, optimizer: LrOptimizer, stride: int = 1, padding: int = 0):
        self.kernels = kernels
        self.kernels_grad = kernels * 0
        self.optimizer = optimizer
        self.stride = stride
        self.padding = padding
        self.out_channels = kernels.shape[0]
        self.in_channels = kernels.shape[1]
        self.kernel_size = kernels.shape[2]

    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        padded = pad3d(inputs, self.padding)
        # channels x height x width

        fragments = extract_fragments3d(padded, self.kernel_size, self.stride)
        # height x width x channels x kernelSize x kernelSize

        reshaped_fragments: np.ndarray = np.expand_dims(fragments, 2)
        # height x width x 1 x channels x kernelSize x kernelSize

        multiplied: np.ndarray = reshaped_fragments * self.kernels
        # height x width x outputChannels x channels x kernelSize x kernelSize

        # Collapse 3 last dimensions into one
        new_shape = [*multiplied.shape]
        new_shape[-3] = new_shape[-3] * new_shape[-2] * new_shape[-1]

        reshaped = multiplied.reshape(new_shape[:-2])
        # height x width x channels x elementsToSum

        summed = reshaped.sum(reshaped.ndim - 1)
        # height x width x channels

        return np.moveaxis(summed, 2, 0)

    def backpropagate_gradient(self, inputs: np.ndarray, outputs: np.ndarray, current_gradient: np.ndarray,
                               config: TrainingConfig):
        padded = pad3d(inputs, self.padding)
        self.apply_gradient(padded, current_gradient, config)

        channels, new_height, new_width = get_dims_after_filter(padded.shape, self.kernel_size, self.stride)
        padded_input_error = np.zeros(padded.shape)

        for c in range(self.out_channels):
            kernel = self.kernels[c]
            for h in range(new_height):
                for w in range(new_width):
                    h_offset = h * self.stride
                    w_offset = w * self.stride

                    padded_input_error[:, h_offset:h_offset + self.kernel_size, w_offset:w_offset + self.kernel_size] \
                        += kernel * current_gradient[c, h, w]

        return remove_padding3d(padded_input_error, self.padding)

    def apply_gradient(self, inputs: np.ndarray, gradient: np.ndarray, config: TrainingConfig):
        factor = 1 / config.batch_size

        for c in range(self.out_channels):
            kernel: np.ndarray = self.kernels[c]
            kernel_grad = np.zeros(kernel.shape)
            shape = inputs.shape

            for ic in range(self.in_channels):
                for h in range(self.kernel_size):
                    for w in range(self.kernel_size):
                        affected: np.ndarray = inputs[ic,
                                               h:shape[1] - (self.kernel_size - h - 1): self.stride,
                                               w:shape[2] - (self.kernel_size - w - 1): self.stride]
                        mean = (gradient[c] * affected).mean()
                        kernel_grad[ic, h, w] = mean

            self.kernels_grad = factor * kernel_grad

    def train(self, config: TrainingConfig):
        optimized = self.optimizer.optimize(self.kernels_grad, config)
        self.kernels += optimized
        self.kernels_grad *= 0
