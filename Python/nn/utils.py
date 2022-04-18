import random
from typing import TypeVar

import numpy as np

epsilon = 1e-7
T = TypeVar("T")


def lerp_arrays(a: np.ndarray, b: np.ndarray, t: float):
    return a + (b - a) * t


def init_random(seed: int = 777):
    np.random.seed(seed)
    random.seed(seed)


def choose_random(elements: list[T]) -> T:
    return elements[random.randint(0, len(elements) - 1)]


def select_random(elements: list, count: int):
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


def split_array(array: list[T], max_size: int):
    sub_arrays = []
    parts = len(array) // max_size

    for i in range(parts):
        sub_arrays.append(array[i * max_size:(i + 1) * max_size])

    if len(array) % max_size != 0:
        sub_arrays.append(array[parts * max_size:])

    return sub_arrays
