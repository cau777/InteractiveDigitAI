from abc import abstractmethod, ABC
from typing import Iterator, Any
from codebase.nn import TrainingConfig

import numpy as np
import timeit


class NNLayer(ABC):
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, object]:
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray, cache: object) -> np.ndarray:
        pass

    def train(self, config: TrainingConfig):
        pass

    @property
    def trainable_params_count(self) -> int:
        return 0

    def get_trainable_params(self) -> list[float]:
        return []

    def set_trainable_params(self, params_iterator: Iterator[float]) -> None:
        pass

    def benchmark_forward(self, inputs: np.ndarray, results: list[tuple[str, float]]) -> tuple[np.ndarray, object]:
        name = self.__class__.__name__
        time = timeit.timeit(lambda: self.forward(inputs), number=10)
        results.append((name, time))
        return self.forward(inputs)

    def benchmark_backward(self, current_gradient: np.ndarray, cache: object,
                           results: list[tuple[str, float]]) -> np.ndarray:
        name = self.__class__.__name__
        time = timeit.timeit(lambda: self.backward(current_gradient, cache), number=10)
        results.append((name, time))
        return self.backward(current_gradient, cache)

    def benchmark_train(self, config: TrainingConfig, results: list[tuple[str, float]]):
        name = self.__class__.__name__
        time = timeit.timeit(lambda: self.train(config), number=10)
        results.append((name, time))
