import numpy as np

from codebase.nn.loss_functions import LossFunction


def softmax(x: np.ndarray):
    e = np.exp(x - np.max(x))
    return e / np.sum(e, axis=-1)


class CrossEntropyLossFunction(LossFunction):
    def calc_loss(self, expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        p = softmax(actual)
        labels = expected.argmax(-1)
        return np.sum(-np.log(p[labels]))

    def calc_loss_gradient(self, expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        soft = softmax(actual)
        soft -= expected
        return -soft
