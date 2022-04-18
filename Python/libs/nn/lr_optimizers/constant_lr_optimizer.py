import numpy as np

from libs.nn.lr_optimizers import LrOptimizer
from libs.nn import TrainingConfig


class ConstantLrOptimizer(LrOptimizer):
    def __init__(self, alpha: float = 0.005):
        self.alpha = alpha

    def optimize(self, gradients: np.ndarray, config: TrainingConfig) -> np.ndarray:
        return gradients * self.alpha


