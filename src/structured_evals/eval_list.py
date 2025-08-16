from typing import Any, Literal

from structured_evals.base import EvaluatorBase


class ListEval(EvaluatorBase[list[Any], float]):
    def __init__(
        self,
        item_evaluator: EvaluatorBase[Any, float],
        aggregation: Literal["average", "sum"] = "average",
    ) -> None:
        super().__init__()
        self.item_evaluator = item_evaluator
        self.aggregation = aggregation

    def evaluate(self, pred: list[Any], target: list[Any]) -> float:
        # what if unequal length?
        #  - some kind of f-score?
        raise NotImplementedError("ListEval not implemented yet")
