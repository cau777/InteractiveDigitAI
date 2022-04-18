from math import sqrt
from typing import Iterator

import numpy as np

from nn import TrainingConfig
from nn.layers import NNLayer
from nn.lr_optimizers import LrOptimizer
from nn.utils import get_dims_after_filter


def pad3d(array: np.ndarray, padding: int):
    height = array.shape[1]
    width = array.shape[2]
    result = np.zeros((array.shape[0], height + 2 * padding, width + 2 * padding))
    result[:, padding:height + padding, padding:width + padding] = array
    return result


def pad4d(array: np.ndarray, padding: int):
    shape = array.shape
    height = shape[-2]
    width = shape[-1]
    result = np.zeros((*shape[:-2], height + 2 * padding, width + 2 * padding))
    result[:, :, padding:height + padding, padding:width + padding] = array
    return result


def remove_padding3d(array: np.ndarray, padding: int):
    shape = array.shape
    return array[:, padding:shape[1] - padding, padding: shape[2] - padding]


def remove_padding4d(array: np.ndarray, padding: int):
    shape = array.shape
    return array[:, :, padding:shape[-2] - padding, padding:shape[-1] - padding]


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


def extract_fragments4d(array: np.ndarray, size: int, stride: int):
    batch, channels, new_height, new_width = get_dims_after_filter(array.shape, size, stride)
    result = np.zeros((new_height, new_width, batch, channels, size, size))
    for h in range(new_height):
        for w in range(new_width):
            h_offset = h * stride
            w_offset = w * stride
            result[h, w] = array[:, :, h_offset:h_offset + size, w_offset:w_offset + size]
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

    @staticmethod
    def create_random(out_channels: int, in_channels: int, kernel_size: int, optimizer: LrOptimizer,
                      stride: int = 1, padding: int = 0):
        if stride < 1:
            raise ValueError("Stride can't be less than 1")
        if padding < 0:
            raise ValueError("Padding can't be negative")

        # 'He normal' initialization
        fan_in = in_channels * kernel_size * kernel_size
        std_dev = sqrt(2 / fan_in)
        kernels = np.random.normal(0, std_dev, (out_channels, in_channels, kernel_size, kernel_size))
        return ConvolutionLayer(kernels, optimizer, stride, padding)

    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        padded = pad4d(inputs, self.padding)
        # batch x in_channels x height x width

        fragments = extract_fragments4d(padded, self.kernel_size, self.stride)
        # height x width x batch x in_channels x kernelSize x kernelSize

        reshaped_fragments: np.ndarray = np.expand_dims(np.moveaxis(fragments, 2, 0), 3)
        # batch x height x width x 1 x in_channels x kernelSize x kernelSize

        multiplied: np.ndarray = reshaped_fragments * self.kernels
        # batch x height x width x out_channels x in_channels x kernelSize x kernelSize

        # Collapse 3 last dimensions into one
        new_shape = [*multiplied.shape]
        new_shape[-3] = new_shape[-3] * new_shape[-2] * new_shape[-1]

        reshaped = multiplied.reshape(new_shape[:-2])
        # batch x height x width x out_channels x elementsToSum

        summed = reshaped.sum(-1)
        # batch x height x width x out_channels

        result = np.moveaxis(summed, 3, 1)
        return result

    def backpropagate_gradient(self, inputs: np.ndarray, outputs: np.ndarray, current_gradient: np.ndarray,
                               config: TrainingConfig):
        padded = pad4d(inputs, self.padding)
        self.apply_gradient(padded, current_gradient, config)

        batch, channels, new_height, new_width = get_dims_after_filter(padded.shape, self.kernel_size, self.stride)
        padded_input_grad = np.zeros(padded.shape)

        for h in range(new_height):
            for w in range(new_width):
                h_offset = h * self.stride
                w_offset = w * self.stride

                for b in range(batch):
                    grad = np.zeros((*self.kernels.shape[-3:],))
                    for c in range(channels):
                        grad += self.kernels[c] * current_gradient[b, c, h, w]
                    padded_input_grad[b, :, h_offset:h_offset + self.kernel_size, w_offset:w_offset + self.kernel_size] = grad

        return remove_padding4d(padded_input_grad, self.padding)

    def apply_gradient(self, inputs: np.ndarray, gradient: np.ndarray, config: TrainingConfig):
        factor = 1
        kernels_grad = np.zeros(self.kernels.shape)
        shape = inputs.shape
        batch_size = shape[0]

        for ic in range(self.in_channels):
            for h in range(self.kernel_size):
                for w in range(self.kernel_size):
                    affected: np.ndarray = inputs[:, ic,
                                           h:shape[-2] - (self.kernel_size - h - 1): self.stride,
                                           w:shape[-1] - (self.kernel_size - w - 1): self.stride]

                    for b in range(batch_size):
                        mean = (gradient[b] * affected[b]).mean()
                        kernels_grad[b, ic, h, w] += mean

        self.kernels_grad += factor * kernels_grad

    def train(self, config: TrainingConfig):
        optimized = self.optimizer.optimize(self.kernels_grad, config)
        self.kernels += optimized
        self.kernels_grad *= 0

    def trainable_params_count(self) -> int:
        return self.kernels.size

    def trainable_params(self) -> list[float]:
        return list(self.kernels.flat)

    def set_trainable_params(self, params_iterator: Iterator[float]) -> None:
        self.kernels = np.array([next(params_iterator) for _ in range(self.kernels.size)]).reshape(self.kernels.shape)
