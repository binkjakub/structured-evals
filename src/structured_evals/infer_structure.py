from datetime import datetime
from typing import Any, Literal

from structured_evals.base import EvaluatorBase
from structured_evals.eval_batch import BatchDictEval
from structured_evals.eval_dict import DictEval
from structured_evals.eval_primitive import DateEval, NumEval
from structured_evals.eval_text import EvalTextualMetric
from structured_evals.ngram_score_fn import chrf_eval

DEFAULT_NGRAM_EVALUATOR = EvalTextualMetric(chrf_eval, "chrf")
DEFAULT_BATCH_AGGREGATION = "average"
DEFAULT_ERROR_STRATEGY: Literal["raise", "ignore"] = "raise"


def infer_structured_evaluator(data: Any) -> EvaluatorBase:
    if isinstance(data, dict):
        return DictEval(
            eval_mapping={key: infer_structured_evaluator(value) for key, value in data.items()}
        )
    elif isinstance(data, str):
        return DEFAULT_NGRAM_EVALUATOR
    elif isinstance(data, float):
        return NumEval()
    elif isinstance(data, int):
        return NumEval()
    elif isinstance(data, datetime):
        return DateEval()
    elif isinstance(data, list):
        return BatchDictEval(
            item_evaluator=infer_structured_evaluator(data[0]),  # type: ignore
            aggregation=DEFAULT_BATCH_AGGREGATION,
            error_strategy=DEFAULT_ERROR_STRATEGY,
        )
    else:
        raise ValueError(f"Unsupported type: {type(data)}")
