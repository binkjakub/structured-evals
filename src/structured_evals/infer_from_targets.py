import datetime
from typing import Any, Literal

from dotenv import dotenv_values
from langchain_openai import ChatOpenAI

from structured_evals.base import EvaluatorBase
from structured_evals.eval_dict import DictEval
from structured_evals.eval_list import ListEval, T_list_aggregation
from structured_evals.eval_llm_as_judge import LlmAsJudge
from structured_evals.eval_primitive import DateEval, NumEval
from structured_evals.eval_text import EvalTextualMetric
from structured_evals.ngram_score_fn import chrf_eval

DEFAULT_BATCH_AGGREGATION = "average"
DEFAULT_LIST_AGGREGATION: T_list_aggregation = "average"
DEFAULT_ERROR_STRATEGY: Literal["raise", "ignore"] = "raise"


def infer_structured_evaluator(
    data: Any,
    text_evaluator: Literal["ngram", "llm"],
) -> EvaluatorBase:
    if isinstance(data, dict):
        assert len(data) > 0, "Dict must not be empty to infer evaluator"
        return DictEval(
            eval_mapping={
                key: infer_structured_evaluator(value, text_evaluator)
                for key, value in data.items()
            }
        )
    elif isinstance(data, str):
        if text_evaluator == "ngram":
            return EvalTextualMetric(chrf_eval, "chrf")
        elif text_evaluator == "llm":
            return get_default_llm_as_judge()
        else:
            raise ValueError(f"Invalid text_evaluator: {text_evaluator}")
    elif isinstance(data, float):
        return NumEval()
    elif isinstance(data, int):
        return NumEval()
    elif isinstance(data, (datetime.datetime, datetime.date)):
        return DateEval()
    elif isinstance(data, list):
        assert len(data) > 0, "List must not be empty to infer evaluator"
        return ListEval(
            item_evaluator=infer_structured_evaluator(data[0], text_evaluator),
            aggregation=DEFAULT_LIST_AGGREGATION,
        )
    else:
        raise ValueError(
            f"Unsupported type encountered during structured evaluator inference: {type(data)}"
        )


def get_default_llm_as_judge() -> LlmAsJudge:
    config = dotenv_values()
    return LlmAsJudge(
        llm=ChatOpenAI(
            model=config["OPENAI_MODEL"],  # type: ignore
            base_url=config["OPENAI_BASE_URL"],  # type: ignore
            api_key=config["OPENAI_API_KEY"],  # type: ignore
        ),
    )
