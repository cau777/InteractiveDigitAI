import numpy as np

from codebase.nn.lr_optimizers import LrOptimizer
from codebase.nn import TrainingConfig
from codebase.nn.utils import lerp_arrays, epsilon


class AdamLrOptimizer(LrOptimizer):
    def __init__(self, alpha: float = 0.001, decay1: float = 0.9, decay2: float = 0.999):
        self.alpha = alpha
        self.decay1 = decay1
        self.decay2 = decay2
        self.moment1: np.ndarray|None = None
        self.moment2: np.ndarray|None = None

    def optimize(self, gradients: np.ndarray, config: TrainingConfig) -> np.ndarray:
        if self.moment1 is None:
            self.moment1 = np.zeros(gradients.shape)
        if self.moment2 is None:
            self.moment2 = np.zeros(gradients.shape)

        self.moment1 = lerp_arrays(gradients, self.moment1, self.decay1)
        self.moment2 = lerp_arrays(np.square(gradients), self.moment2, self.decay2)

        moment1_b = self.moment1 / (1 - (self.decay1 ** config.epoch))
        moment2_b = self.moment2 / (1 - (self.decay2 ** config.epoch))

        return self.alpha * moment1_b / (np.sqrt(moment2_b) + epsilon)


