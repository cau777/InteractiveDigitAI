from abc import abstractmethod
from typing import TypeVar, Callable, Iterator, Sequence, overload

T = TypeVar("T")
TFrom = TypeVar("TFrom")


class LazyList(Sequence[T]):
    def __init__(self, values: list[TFrom], converter: Callable[[TFrom], T]):
        self.values = values
        self.converter = converter
        self.cached = dict()

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[T]:
        return map(lambda i: self[i], range(len(self)))

    def __contains__(self, item):
        return any(map(lambda x: x == item, iter(self)))

    @overload
    @abstractmethod
    def __getitem__(self, i: int) -> T:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, s: slice) -> Sequence[T]:
        ...

    def __getitem__(self, i: int | slice) -> T:
        if isinstance(i, int):
            if i in self.cached:
                return self.cached[i]

            value = self.converter(self.values[i])
            self.cached[i] = value
            return value
        elif isinstance(i, slice):
            return [self[index] for index in range(*i.indices(len(self)))]

    def __eq__(self, other):
        return all(map(lambda x, y: x == y, self, other))
