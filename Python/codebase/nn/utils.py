import random
from typing import TypeVar, Sequence

import numpy as np

epsilon = 1e-7
T = TypeVar("T")


def lerp_arrays(a: np.ndarray, b: np.ndarray, t: float):
    return a + (b - a) * t


def init_random(seed: int = 777):
    np.random.seed(seed)
    random.seed(seed)


def choose_random(elements: Sequence[T]) -> T:
    return elements[random.randint(0, len(elements) - 1)]


def select_random(elements: Sequence, count: int):
    result = list()
    for i in range(count):
        result.append(choose_random(elements))
    return result


def get_dims_after_filter(shape: tuple[int, ...], size: int, stride: int):
    return (
        *shape[:-2],
        (shape[-2] - size) // stride + 1,
        (shape[-1] - size) // stride + 1
    )
