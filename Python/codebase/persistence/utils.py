import zlib
import numpy as np
from struct import Struct
from typing import Sequence
from codebase.general_utils import split_array, get_size
from codebase.nn.training_example import ClassificationExample
from codebase.persistence import LazyList

DEFAULT_COMPRESSION = 6


class ClassificationPattern:
    def __init__(self, inputs_shape: tuple[int, ...], classes: int):
        self.inputs_shape = inputs_shape
        self.inputs_size = get_size(inputs_shape)
        self.classes = classes
        self.struct = Struct(("f" * self.inputs_size) + "I")


CLASSIFICATION_PATTERNS: dict[str, ClassificationPattern] = {
    "mnist": ClassificationPattern((1, 28, 28), 10)
}


def load_classification(data: bytes, pattern: ClassificationPattern):
    parts = pattern.struct.unpack(data)
    return ClassificationExample(np.array(parts[:-1]).reshape(pattern.inputs_shape), parts[-1], pattern.classes)


def save_classification(obj: ClassificationExample, pattern: ClassificationPattern):
    return pattern.struct.pack(*tuple(obj.inputs.flat), obj.label_class)


def load_compressed_classification(pattern_name: str, data: bytes) -> LazyList[ClassificationExample]:
    print(f"Decompressing {len(data)} bytes")
    decompressed = zlib.decompress(data)
    pattern = CLASSIFICATION_PATTERNS[pattern_name]
    slices = split_array(decompressed, pattern.struct.size)
    return LazyList(slices, lambda x: load_classification(x, pattern))


def save_compressed_classification(pattern_name: str, examples: Sequence[ClassificationExample]):
    pattern = CLASSIFICATION_PATTERNS[pattern_name]
    b = b"".join([save_classification(obj, pattern) for obj in examples])
    return zlib.compress(b, DEFAULT_COMPRESSION)
