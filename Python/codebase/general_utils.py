from typing import TypeVar, Sequence

T = TypeVar("T")


def get_size(shape: tuple[int, ...]):
    size = 1
    for num in shape:
        size *= num
    return size


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
