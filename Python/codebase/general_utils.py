from typing import TypeVar, Sequence, Iterator, Any

import numpy as np

T = TypeVar("T")


def split_array(array: Sequence[T], max_size: int):
    sub_arrays = []
    parts = len(array) // max_size

    for i in range(parts):
        sub_arrays.append(array[i * max_size:(i + 1) * max_size])

    if len(array) % max_size != 0:
        sub_arrays.append(array[parts * max_size:])

    return sub_arrays


def product(*elements: Sequence[int]):
    result = 1
    for e in elements:
        result *= e
    return result


def take_iter(iterator: Iterator[T], count: int):
    return map(lambda index, val: val, range(count), iterator)


def to_one_hot_vec(num: int, total: int):
    vec = [0] * total
    vec[num] = 1
    return vec


def from_one_hot_vec(vec: list[float]):
    for index, element in enumerate(vec):
        if abs(1 - element) < 0.001:
            return index
    raise ValueError(vec)


def to_flat_list(arr: np.ndarray) -> list[float]:
    result: Any = arr.flatten().tolist()
    return result
