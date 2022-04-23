from abc import abstractmethod, ABC
from typing import Iterator
from codebase.nn import TrainingConfig

import numpy as np
import timeit


class NNLayer(ABC):
    @abstractmethod
    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backpropagate_gradient(self, inputs: np.ndarray, outputs: np.ndarray, current_gradient: np.ndarray,
                               config: TrainingConfig) -> np.ndarray:
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

    def benchmark_feed_forward(self, inputs: np.ndarray, results: list[tuple[str, float]]) -> np.ndarray:
        name = self.__class__.__name__
        time = timeit.timeit(lambda: self.feed_forward(inputs), number=10)
        results.append((name, time))
        return self.feed_forward(inputs)

    def benchmark_backprapagate(self, inputs: np.ndarray, outputs: np.ndarray, current_gradient: np.ndarray,
                                config: TrainingConfig, results: list[tuple[str, float]]) -> np.ndarray:
        name = self.__class__.__name__
        time = timeit.timeit(lambda: self.backpropagate_gradient(inputs, outputs, current_gradient, config), number=10)
        results.append((name, time))
        return self.backpropagate_gradient(inputs, outputs, current_gradient, config)

    def benchmark_train(self, config: TrainingConfig, results: list[tuple[str, float]]):
        name = self.__class__.__name__
        time = timeit.timeit(lambda: self.train(config), number=10)
        results.append((name, time))
