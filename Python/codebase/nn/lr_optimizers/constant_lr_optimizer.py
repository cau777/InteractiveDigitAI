import numpy as np

from codebase.nn.lr_optimizers import LrOptimizer
from codebase.nn import BatchConfig


class ConstantLrOptimizer(LrOptimizer):
    def __init__(self, alpha: float = 0.005):
        self.alpha = alpha

    def optimize(self, gradients: np.ndarray, config: BatchConfig) -> np.ndarray:
        return gradients * self.alpha


