from typing import Collection

from structured_evals.base import EvaluatorBase, ItemEvalOutput, T_in

T_enum = str | int | float | None


class EnumItemOutput(ItemEvalOutput):
    prohibited_value: int


class EnumEval(EvaluatorBase[T_enum, EnumItemOutput]):
    def __init__(self, allowed_values: Collection[T_enum], name: str | None = None) -> None:
        super().__init__(name)
        self.allowed_values = set(allowed_values)

    @property
    def zero_score(self) -> EnumItemOutput:
        return EnumItemOutput(score=0.0, prohibited_value=0)

    @property
    def max_score(self) -> EnumItemOutput:
        return EnumItemOutput(score=1.0, prohibited_value=0)

    def evaluate(self, pred: T_enum, target: T_enum) -> EnumItemOutput:
        if self.is_null(pred) and self.is_null(target):
            return EnumItemOutput(score=1.0, prohibited_value=0)
        if not self.check_dtype(pred, target):
            return EnumItemOutput(score=0.0, prohibited_value=0)

        pred_prohibited = int(pred not in self.allowed_values)

        if pred in self.allowed_values and target in self.allowed_values and pred == target:
            return EnumItemOutput(score=1.0, prohibited_value=0)

        return EnumItemOutput(score=0.0, prohibited_value=pred_prohibited)

    def is_null(self, item: T_enum) -> bool:
        return item is None

    def check_dtype(self, pred: T_in, target: T_in) -> bool:
        return isinstance(pred, (str, int, float, type(None))) and isinstance(
            target, (str, int, float, type(None))
        )
