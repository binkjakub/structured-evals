from typing import Callable

from sevals.base import Evaluator, T_out


class EvalTextualMetric(Evaluator[str, T_out]):
    def __init__(self, metric_fn: Callable[[str, str], T_out], metric_name: str):
        super().__init__(metric_name)
        self.metric_fn = metric_fn

    def evaluate(self, pred: str, target: str) -> T_out:
        return self.metric_fn(pred, target)