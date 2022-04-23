import zlib
from typing import TypeVar, Type

DEFAULT_COMPRESSION = 4
T = TypeVar("T")


def load_compressed(cls: Type[T], data: bytes) -> T:
    print(f"Decompressing {len(data)} bytes")
    decompressed = zlib.decompress(data)

    print(f"Parsing {len(decompressed)} bytes")
    obj = cls()
    obj.ParseFromString(decompressed)
    return obj


def save_compressed(obj):
    b: bytes = obj.SerializeToString()
    return zlib.compress(b, DEFAULT_COMPRESSION)
