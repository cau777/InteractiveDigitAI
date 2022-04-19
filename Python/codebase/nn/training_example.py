import numpy as np
from codebase.nn.utils import epsilon


def create_classification_example(inputs: np.ndarray, label: int, total_classes: int):
    label_array = np.zeros(total_classes).reshape(total_classes, 1) + epsilon
    label_array[label, 0] = 1 - epsilon
    return TrainingExample(inputs, label_array)


class TrainingExample:
    def __init__(self, inputs: np.ndarray, label: np.ndarray):
        self.inputs = inputs
        self.label = label
