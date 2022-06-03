import numpy as np

from codebase.nn import BatchConfig
from codebase.nn.layers import NNLayer

Cache = tuple[list, list[int]]


class StackLayer(NNLayer):
    def __init__(self, *layers: NNLayer):
        self.layers = list(layers)

    def forward(self, inputs: np.ndarray, config: BatchConfig) -> tuple[np.ndarray, Cache]:
        caches: list = [None] * len(self.layers)
        outputs = []
        dims = []

        for i in range(len(self.layers)):
            output, caches[i] = self.layers[i].forward(inputs, config)
            outputs.append(output)
            dims.append(output.shape[1])

        stacked = np.hstack(outputs)
        return stacked, (caches, np.cumsum(dims))

    def backward(self, grad: np.ndarray, cache: Cache, config: BatchConfig) -> np.ndarray:
        caches, sections = cache
        split = np.hsplit(grad, sections)
        result = None

        for layer, layer_cache, layer_grad in zip(self.layers, caches, split):
            layer_result = layer.backward(layer_grad, layer_cache, config)
            if result is None:
                result = layer_result
            else:
                result += layer_result

        return result

    def train(self, config: BatchConfig):
        for layer in self.layers:
            layer.train(config)
