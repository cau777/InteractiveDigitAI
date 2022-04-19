from abc import ABC, abstractmethod

import numpy as np


class MeasureTracker(ABC):
    @abstractmethod
    def track(self, inputs: np.ndarray, outputs: np.ndarray, label: np.ndarray, loss: np.ndarray) -> None:
        pass

    @abstractmethod
    def record(self, d: dict[str]):
        pass


class AvgLossTracker(MeasureTracker):
    def __init__(self):
        self.count = 0
        self.loss = 0

    def track(self, inputs: np.ndarray, outputs: np.ndarray, label: np.ndarray, loss: np.ndarray):
        self.count += 1
        self.loss += np.abs(loss).sum() / loss.size

    def record(self, d: dict[str]):
        d["avg_loss"] = self.loss / self.count
        self.count = 0
        self.loss = 0


class AccuracyTracker(MeasureTracker):
    def __init__(self):
        self.count = 0
        self.right = 0

    def track(self, inputs: np.ndarray, outputs: np.ndarray, label: np.ndarray, loss: np.ndarray):
        self.count += 1
        if label.argmax() == outputs.argmax():
            self.right += 1

    def record(self, d: dict[str]):
        d["accuracy"] = self.right / self.count
        self.count = 0
        self.right = 0


def create_tracker(name: str) -> MeasureTracker:
    if name == "avg_loss":
        return AvgLossTracker()
    if name == "accuracy":
        return AccuracyTracker()

    raise ValueError(name + " is not a valid tracker")
