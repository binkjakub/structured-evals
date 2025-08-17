from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar

T_in = TypeVar("T_in")
T_out = TypeVar("T_out")
ErrorStrategy = Literal["raise", "ignore"]


class EvaluatorBase(ABC, Generic[T_in, T_out]):
    def __init__(self, name: str | None = None) -> None:
        self.__name = name

    def __call__(self, pred: T_in, target: T_in) -> T_out:
        self.check_dtype(pred, target)
        return self.evaluate(pred, target)

    @abstractmethod
    def check_dtype(self, pred: T_in, target: T_in) -> None:
        pass

    @abstractmethod
    def evaluate(self, pred: T_in, target: T_in) -> T_out:
        pass

    @property
    def name(self) -> str:
        return self.__name or self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
