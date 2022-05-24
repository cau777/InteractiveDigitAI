import numpy as np

from codebase.nn.layers import NNLayer


class DebugLayer(NNLayer):
    def __init__(self, tag):
        self.tag = tag

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, None]:
        print(f"{self.tag} - Forward: inputs.shape={inputs.shape}")
        return inputs, None

    def backward(self, grad: np.ndarray, cache: None) -> np.ndarray:
        print(f"{self.tag} - Backward: grad.shape={grad.shape}")
        return grad
