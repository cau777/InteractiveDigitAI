import numpy as np

from nn.lr_optimizers.lr_optimizer import LrOptimizer
from nn.training_config import TrainingConfig


class ConstantLrOptimizer(LrOptimizer):
    def __init__(self, alpha: float = 0.005):
        self.alpha = alpha

    def optimize(self, gradients: np.ndarray, config: TrainingConfig) -> np.ndarray:
        return gradients * self.alpha


