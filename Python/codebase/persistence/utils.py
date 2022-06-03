import zlib
from abc import ABC
from struct import Struct
from typing import Sequence

import numpy as np

from codebase.general_utils import split_array, to_one_hot_vec, from_one_hot_vec
from codebase.general_utils import to_flat_list
from codebase.nn.prediction_example import PredictionExample
from codebase.persistence import LazyList

DEFAULT_COMPRESSION = 6


class Pattern(ABC):
    def __init__(self, shape: tuple[int, ...], params: int, one_hot_vectors: list[int]):
        self.shape = shape
        self.one_hot_vectors = one_hot_vectors
        self.model = Struct(("I" * len(one_hot_vectors)) + ("f" * params))

    def __len__(self):
        return self.model.size

    def from_bytes(self, b: bytes):
        values = self.model.unpack(b)

        array = []
        index = 0
        for vec_size in self.one_hot_vectors:
            num = values[index]
            array.extend(to_one_hot_vec(num, vec_size))
            index += 1

        array.extend(values[index:])
        return np.array(array, dtype="float32").reshape(self.shape)

    def to_bytes(self, data: np.ndarray) -> bytes:
        array = to_flat_list(data)
        values = []
        index = 0
        for vec_size in self.one_hot_vectors:
            values.append(from_one_hot_vec(array[index: index + vec_size]))
            index += vec_size
        values.extend(array[index:])
        return self.model.pack(*values)


PATTERNS: dict[str, tuple[Pattern, Pattern]] = {
    "mnist": (Pattern((1, 28, 28), 784, []), Pattern((10,), 0, [10]))
}


def load(data: bytes, pattern: tuple[Pattern, Pattern]):
    inputs_pattern, outputs_pattern = pattern
    sep = len(inputs_pattern)
    return PredictionExample(inputs_pattern.from_bytes(data[:sep]), outputs_pattern.from_bytes(data[sep:]))


def save(obj: PredictionExample, pattern: tuple[Pattern, Pattern]):
    return pattern[0].to_bytes(obj.inputs) + pattern[1].to_bytes(obj.label)


def load_compressed(pattern_name: str, data: bytes):
    print(f"Decompressing {len(data)} bytes")
    decompressed = zlib.decompress(data)
    pattern = PATTERNS[pattern_name]
    slices = split_array(decompressed, len(pattern[0]) + len(pattern[1]))
    return LazyList(slices, lambda x: load(x, pattern))


def save_compressed(pattern_name: str, examples: Sequence[PredictionExample]):
    pattern = PATTERNS[pattern_name]
    b = b"".join([save(obj, pattern) for obj in examples])
    return zlib.compress(b, DEFAULT_COMPRESSION)
