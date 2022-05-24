import numpy as np


def arrays_equal(arr1: np.ndarray, arr2: np.ndarray, tolerance: float):
    delta = np.abs(arr1 - arr2)
    return np.all(delta <= tolerance)
