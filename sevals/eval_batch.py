from dataclasses import dataclass
from typing import Any, Literal

from sevals.base import Evaluator
from sevals.eval_dict import DictEval, DictEvalOutput


@dataclass(kw_only=True)
class EvalBatchOutput:
    item_results: list[DictEvalOutput]
    agg_results: DictEvalOutput


class EvalBatchDict(Evaluator[list[dict[str, Any]], EvalBatchOutput]):
    def __init__(self, item_evaluator: DictEval, aggregation: Literal["average"]) -> None:
        super().__init__()
        self.item_evaluator = item_evaluator
        self.aggregation = aggregation

    def evaluate(self, pred: list[dict[str, Any]], target: list[dict[str, Any]]) -> EvalBatchOutput:
        item_results = [self.item_evaluator.evaluate(p, t) for p, t in zip(pred, target)]
        agg_results = DictEvalOutput.aggregate(item_results, self.aggregation)
        return EvalBatchOutput(item_results=item_results, agg_results=agg_results)
