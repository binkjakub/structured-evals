from typing import Any, Literal

from dotenv import dotenv_values
from langchain_openai import ChatOpenAI

from structured_evals.base import EvaluatorBase
from structured_evals.eval_dict import DictEval
from structured_evals.eval_enum import EnumEval
from structured_evals.eval_list import ListEval, T_list_aggregation
from structured_evals.eval_llm_as_judge import LlmAsJudge
from structured_evals.eval_primitive import DateEval, NumEval
from structured_evals.eval_text import EvalTextualMetric
from structured_evals.ngram_score_fn import chrf_eval

DEFAULT_BATCH_AGGREGATION = "average"
DEFAULT_LIST_AGGREGATION: T_list_aggregation = "average"
DEFAULT_ERROR_STRATEGY: Literal["raise", "ignore"] = "raise"


def infer_structured_evaluator_from_schema(
    schema: dict[str, Any],
    text_evaluator: Literal["ngram", "llm"],
) -> EvaluatorBase:
    if isinstance(schema, dict):
        assert len(schema) > 0, "Schema must not be empty to infer evaluator"
        return DictEval(
            eval_mapping={
                key: _infer_evaluator(item_schema, text_evaluator)
                for key, item_schema in schema.items()
            }
        )
    raise ValueError(f"Unsupported schema type: {type(schema)}")


def _infer_evaluator(
    item_schema: dict[str, Any], text_evaluator: Literal["ngram", "llm"]
) -> EvaluatorBase:
    assert "type" in item_schema, "Schema must contain 'type' key"

    if item_schema["type"] == "string":
        if item_schema.get("format") == "date":
            # handles case when schema is compatible with json_schema
            return DateEval()
        elif text_evaluator == "ngram":
            return EvalTextualMetric(chrf_eval, "chrf")
        elif text_evaluator == "llm":
            return get_default_llm_as_judge()
        else:
            raise ValueError(f"Invalid text_evaluator: {text_evaluator}")
    elif item_schema["type"] == "date":
        return DateEval()
    elif item_schema["type"] in ["integer", "float", "number"]:
        return NumEval()
    elif item_schema["type"] == "enum":
        return EnumEval(item_schema["choices"])
    elif item_schema["type"] in ["array", "list"]:
        assert len(item_schema["items"]) > 0, "List must not be empty to infer evaluator"
        return ListEval(
            item_evaluator=_infer_evaluator(item_schema["items"], text_evaluator),
            aggregation=DEFAULT_LIST_AGGREGATION,
        )
    else:
        raise ValueError(
            f"Unsupported type encountered during structured evaluator inference: {item_schema['type']}"
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
