import numpy as np

from codebase.general_utils import to_one_hot_vec


class PredictionExample:
    def __init__(self, inputs: np.ndarray, label: np.ndarray):
        self.inputs = inputs.astype("float32")
        self.label = label.astype("float32")

    def __str__(self):
        return f"PredictionExample({self.inputs}, {self.label})"

    def __repr__(self):
        return self.__str__()


def create_classification_example(inputs: np.ndarray, label: int, total_classes: int):
    return PredictionExample(inputs, np.array(to_one_hot_vec(label, total_classes)))

