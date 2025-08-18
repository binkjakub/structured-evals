from typing import Any, Callable

from structured_evals.base import EvaluatorBase, ItemEvalOutput


class EvalTextualMetric(EvaluatorBase[str, ItemEvalOutput]):
    def __init__(self, metric_fn: Callable[[str, str], float], metric_name: str):
        super().__init__(metric_name)
        self.metric_fn = metric_fn

    @property
    def zero_score(self) -> ItemEvalOutput:
        return ItemEvalOutput(score=0.0)

    @property
    def max_score(self) -> ItemEvalOutput:
        return ItemEvalOutput(score=1.0)

    def evaluate(self, pred: str | None, target: str | None) -> ItemEvalOutput:
        if self.is_null(pred) and self.is_null(target):
            return ItemEvalOutput(score=1.0)
        elif self.is_null(pred) or self.is_null(target):
            return ItemEvalOutput(score=0.0)
        elif not self.check_dtype(pred, target):
            return ItemEvalOutput(score=0.0)

        assert isinstance(pred, str) and isinstance(target, str)
        return ItemEvalOutput(score=self.metric_fn(pred, target))

    def is_null(self, item: str | None) -> bool:
        return item is None or item == ""

    def check_dtype(self, pred: Any, target: Any) -> bool:
        return isinstance(pred, str) and isinstance(target, str)
