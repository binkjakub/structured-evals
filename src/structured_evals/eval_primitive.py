import datetime

from structured_evals.base import EvaluatorBase, ItemEvalOutput, T_in

T_numeric = int | float | None
T_date = datetime.datetime | datetime.date | None


class NumEval(EvaluatorBase[T_numeric, ItemEvalOutput]):
    # TODO: add support for precision specification
    def evaluate(self, pred: T_numeric, target: T_numeric) -> ItemEvalOutput:
        if not self.check_dtype(pred, target):
            return ItemEvalOutput(score=0.0)
        return ItemEvalOutput(score=float(pred == target))

    def check_dtype(self, pred: T_in, target: T_in) -> bool:
        return isinstance(pred, T_numeric) and isinstance(target, T_numeric)


class DateEval(EvaluatorBase[T_date, ItemEvalOutput]):
    def __init__(self, date_fmt: str = "%Y-%m-%d") -> None:
        super().__init__()
        self.date_fmt = date_fmt

    def evaluate(self, pred: T_date, target: T_date) -> ItemEvalOutput:
        if self.is_null(pred) and self.is_null(target):
            return ItemEvalOutput(score=1.0)
        if not self.check_dtype(pred, target):
            return ItemEvalOutput(score=0.0)

        assert isinstance(pred, (datetime.datetime, datetime.date)) and isinstance(
            target, (datetime.datetime, datetime.date)
        )
        return ItemEvalOutput(
            score=float(pred.strftime(self.date_fmt) == target.strftime(self.date_fmt))
        )

    def is_null(self, item: T_date) -> bool:
        return item is None

    def check_dtype(self, pred: T_in, target: T_in) -> bool:
        return isinstance(pred, (datetime.datetime, datetime.date)) and isinstance(
            target, (datetime.datetime, datetime.date)
        )
