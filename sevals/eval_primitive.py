import datetime

from sevals.base import Evaluator

T_numeric = int | float
T_date = datetime.datetime | datetime.date


class NumEval(Evaluator[T_numeric, float]):
    def evaluate(self, pred: T_numeric, target: T_numeric) -> float:
        return float(pred == target)


class DateEval(Evaluator[T_date, float]):
    def __init__(self, date_fmt: str = "%Y-%m-%d") -> None:
        super().__init__()
        self.date_fmt = date_fmt

    def evaluate(self, pred: T_date, target: T_date) -> float:
        return float(pred.strftime(self.date_fmt) == target.strftime(self.date_fmt))
