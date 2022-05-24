import numpy as np
from codebase.nn import BatchConfig
from abc import ABC, abstractmethod


class LrOptimizer(ABC):
    @abstractmethod
    def optimize(self, gradients: np.ndarray, config: BatchConfig) -> np.ndarray:
        pass
