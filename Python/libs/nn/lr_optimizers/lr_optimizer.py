import numpy as np
from libs.nn import TrainingConfig
from abc import ABC, abstractmethod


class LrOptimizer(ABC):
    @abstractmethod
    def optimize(self, gradients: np.ndarray, config: TrainingConfig) -> np.ndarray:
        pass
