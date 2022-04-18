from typing import Iterator

import numpy as np
from abc import abstractmethod, ABC

from nn import TrainingConfig


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

    @property
    def trainable_params(self) -> list[float]:
        return []

    def set_trainable_params(self, params_iterator: Iterator[float]) -> None:
        pass
