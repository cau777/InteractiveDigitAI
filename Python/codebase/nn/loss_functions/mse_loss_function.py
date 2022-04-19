import numpy as np

from codebase.nn.loss_functions import LossFunction


class MseLossFunction(LossFunction):
    def calc_loss(self, expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return np.square(expected - actual)

    def calc_loss_gradient(self, expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return expected - actual
