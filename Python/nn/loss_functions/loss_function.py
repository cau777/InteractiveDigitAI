from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def calc_loss(self, expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def calc_loss_gradient(self, expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        pass
