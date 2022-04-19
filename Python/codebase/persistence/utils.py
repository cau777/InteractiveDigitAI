import zlib
from typing import TypeVar, Type

DEFAULT_COMPRESSION = 6
T = TypeVar("T")


def load_compressed(cls: Type[T], data: bytes) -> T:
    decompressed = zlib.decompress(data)
    obj = cls()
    obj.ParseFromString(decompressed)
    return obj


def save_compressed(obj):
    b: bytes = obj.SerializeToString()
    return zlib.compress(b, DEFAULT_COMPRESSION)
