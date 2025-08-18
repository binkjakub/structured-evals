from typing import Any

from pydantic import BaseModel

from structured_evals import DictEvalOutput
from structured_evals.aggregations import Aggregation
from structured_evals.eval_batch import BatchDictEvalOutput


class EvaluationReport(BaseModel):
    num_items: int
    aggregated_scores: dict[str, Any]
    raw_scores: list[DictEvalOutput]

    @classmethod
    def from_batch_dict_eval_output(
        cls, outs: BatchDictEvalOutput, aggregation: Aggregation
    ) -> "EvaluationReport":
        return cls(
            num_items=outs.num_items,
            aggregated_scores=aggregation(outs),
            raw_scores=outs.item_results,
        )
