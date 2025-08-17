from typing import Any, Literal

from pydantic import BaseModel

from structured_evals.aggregations import get_aggregation
from structured_evals.base import EvaluatorBase
from structured_evals.eval_dict import DictEval, DictEvalOutput


class BatchDictEvalOutput(BaseModel):
    agg_results: dict[str, Any]
    item_results: list[DictEvalOutput]


class BatchDictEval(EvaluatorBase[list[dict[str, Any]], BatchDictEvalOutput]):
    def __init__(
        self,
        item_evaluator: DictEval,
        aggregation: str,
        error_strategy: Literal["raise", "ignore"] = "raise",
    ) -> None:
        super().__init__()
        self.item_evaluator = item_evaluator
        self.aggregation = get_aggregation(aggregation)
        self.error_strategy = error_strategy

    def evaluate(
        self, pred: list[dict[str, Any]], target: list[dict[str, Any]]
    ) -> BatchDictEvalOutput:
        item_results: list[DictEvalOutput] = []
        for pred_item, target_item in zip(pred, target):
            try:
                item_results.append(self.item_evaluator.evaluate(pred_item, target_item))
            except TypeError as err:
                if self.error_strategy == "raise":
                    raise err
                else:
                    # TODO: handle ignore error strategy
                    raise NotImplementedError(
                        "ignore error strategy not implemented yet (need to define null values)"
                    )
        agg_results = self.aggregation(outs=item_results)
        return BatchDictEvalOutput(item_results=item_results, agg_results=agg_results)

    def check_dtype(self, pred: list[dict[str, Any]], target: list[dict[str, Any]]) -> None:
        if not isinstance(pred, list) or not isinstance(target, list):
            raise ValueError("Both pred and target must be lists.")
        if len(pred) != len(target):
            raise ValueError("Length of pred and target must be the same.")

    def __repr__(self) -> str:
        return (
            f"BatchDictEval(item_evaluator={self.item_evaluator.name}, aggregation={self.aggregation}, error_strategy={self.error_strategy})"
            + "\n"
            + f"\t{self.item_evaluator.__repr__()}"
        )
