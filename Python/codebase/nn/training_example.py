import numpy as np
from codebase.nn.utils import epsilon


class TrainingExample:
    def __init__(self, inputs: np.ndarray, label: np.ndarray):
        self.inputs = inputs
        self.label = label


class ClassificationExample(TrainingExample):
    def __init__(self, inputs: np.ndarray, label: int, total_classes: int):
        label_array = np.zeros(total_classes) + epsilon
        label_array[label] = 1 - epsilon
        self.label_class = label
        super().__init__(inputs, label_array)
