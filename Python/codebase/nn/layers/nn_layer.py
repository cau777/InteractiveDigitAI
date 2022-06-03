import timeit
import numpy as np

from abc import abstractmethod, ABC
from typing import Iterator
from codebase.nn import BatchConfig


class NNLayer(ABC):
    @abstractmethod
    def forward(self, inputs: np.ndarray, config: BatchConfig) -> tuple[np.ndarray, object]:
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray, cache: object, config: BatchConfig) -> np.ndarray:
        pass

    def train(self, config: BatchConfig):
        pass

    def count_trainable_params(self) -> int:
        return 0

    def get_trainable_params(self) -> list[float]:
        return []

    def set_trainable_params(self, params_iterator: Iterator[float]) -> None:
        pass

    def benchmark_forward(self, inputs: np.ndarray, results: list[tuple[str, float]], config: BatchConfig) \
            -> tuple[np.ndarray, object]:
        name = self.__class__.__name__
        time = timeit.timeit(lambda: self.forward(inputs, config), number=10)
        results.append((name, time))
        return self.forward(inputs, config)

    def benchmark_backward(self, current_gradient: np.ndarray, cache: object,
                           results: list[tuple[str, float]], config: BatchConfig) -> np.ndarray:
        name = self.__class__.__name__
        time = timeit.timeit(lambda: self.backward(current_gradient, cache, config), number=10)
        results.append((name, time))
        return self.backward(current_gradient, cache, config)

    def benchmark_train(self, config: BatchConfig, results: list[tuple[str, float]]):
        name = self.__class__.__name__
        time = timeit.timeit(lambda: self.train(config), number=10)
        results.append((name, time))
