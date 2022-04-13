import numpy as np
from abc import abstractmethod, ABC

from nn.training_config import TrainingConfig


class NNLayer(ABC):
    @abstractmethod
    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backpropagate_gradient(self, inputs: np.ndarray, outputs: np.ndarray, current_gradient: np.ndarray,
                               config: TrainingConfig):
        pass

    def train(self, config: TrainingConfig):
        pass
