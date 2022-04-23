import numpy as np
# noinspection PyUnresolvedReferences
from pyodide.console import Console
from codebase.nn import TrainingExample


class ClientInterfaceBase:
    def __init__(self):
        self.console = Console()

    @staticmethod
    def extract_bytes(s: str):
        return s.encode("latin-1")

    def load_classification_example(self, example, classes: int):
        inputs = np.array(list(example.inputs.values)).reshape(tuple(example.inputs.shape))
        labels = np.zeros(classes)
        labels[example.cls] = 1.0
        return TrainingExample(inputs, labels)
