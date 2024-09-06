from abc import ABC, abstractmethod
from typing import Generic

from typing_extensions import TypeVar

T_in = TypeVar("T_in")
T_out = TypeVar("T_out")


class Evaluator(ABC, Generic[T_in, T_out]):
    def __init__(self, name: str | None = None) -> None:
        self.__name = name

    @abstractmethod
    def evaluate(self, pred: T_in, target: T_in) -> T_out:
        pass

    @property
    def name(self) -> str:
        return self.__name or self.__class__.__name__
