import datetime

from structured_evals.base import EvaluatorBase, T_in

T_numeric = int | float
T_date = datetime.datetime | datetime.date


class NumEval(EvaluatorBase[T_numeric, float]):
    # TODO: add support for precision specification
    def evaluate(self, pred: T_numeric, target: T_numeric) -> float:
        return float(pred == target)

    def check_dtype(self, pred: T_in, target: T_in) -> None:
        if not isinstance(pred, T_numeric) or not isinstance(target, T_numeric):
            raise TypeError(f"Both pred and target must be numeric: {T_numeric}.")


class DateEval(EvaluatorBase[T_date, float]):
    def __init__(self, date_fmt: str = "%Y-%m-%d") -> None:
        super().__init__()
        self.date_fmt = date_fmt

    def evaluate(self, pred: T_date, target: T_date) -> float:
        return float(pred.strftime(self.date_fmt) == target.strftime(self.date_fmt))

    def check_dtype(self, pred: T_in, target: T_in) -> None:
        if not isinstance(pred, T_date) or not isinstance(target, T_date):
            raise TypeError(f"Both pred and target must be dates: {T_date}.")
