import numpy as np

from codebase.nn.loss_functions import LossFunction


def softmax(x: np.ndarray):
    e = np.exp(x - np.max(x))
    return e / np.expand_dims(np.sum(e, -1), 1)


class CrossEntropyLossFunction(LossFunction):
    def calc_loss(self, expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        batch = actual.shape[0]
        p = softmax(actual)
        labels = expected.argmax(-1)
        return -np.log(p[range(batch), labels])

    def calc_loss_gradient(self, expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        soft = softmax(actual)
        soft -= expected
        return -soft
