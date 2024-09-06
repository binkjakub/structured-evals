from dataclasses import dataclass
from typing import Any, Literal

from sevals.base import EvaluatorBase
from sevals.eval_dict import DictEval, DictEvalOutput


@dataclass(kw_only=True)
class BatchDictEvalOutput:
    item_results: list[DictEvalOutput]
    agg_results: DictEvalOutput


class BatchDictEval(EvaluatorBase[list[dict[str, Any]], BatchDictEvalOutput]):
    def __init__(
        self,
        item_evaluator: DictEval,
        aggregation: Literal["average"],
        error_strategy: Literal["raise", "ignore"] = "raise",
    ) -> None:
        super().__init__()
        self.item_evaluator = item_evaluator
        self.aggregation = aggregation
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
                    # todo: handle ignore error strategy
                    raise NotImplementedError(
                        "ignore error strategy not implemented yet (need to define null values)"
                    )
        agg_results = DictEvalOutput.aggregate(item_results, self.aggregation)
        return BatchDictEvalOutput(item_results=item_results, agg_results=agg_results)

    def check_dtype(self, pred: list[dict[str, Any]], target: list[dict[str, Any]]) -> None:
        if not isinstance(pred, list) or not isinstance(target, list):
            raise ValueError("Both pred and target must be lists.")
        if len(pred) != len(target):
            raise ValueError("Length of pred and target must be the same.")
