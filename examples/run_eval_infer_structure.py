import json
from pathlib import Path
from pprint import pprint

import yaml
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from loguru import logger

from structured_evals import (
    EvaluationBatch,
    infer_structured_evaluator_from_predictions,
    infer_structured_evaluator_from_schema,
)
from structured_evals.aggregations import AverageAggregation
from structured_evals.eval_batch import BatchDictEval
from structured_evals.eval_dict import DictEval
from structured_evals.report import EvaluationReport

SCHEMA_FILE = "data/franc_loans_schema.yaml"
PRED_FILE = "data/sample_franc_loans.json"
SAVE_FILE = ".local/results.json"

set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))

logger.info("Loading data")
eval_batch = EvaluationBatch.from_json(
    path=PRED_FILE,
    record_format="json",
    pred_key="answer",
    target_key="gold",
)


def infer_from_raw_predictions() -> DictEval:
    logger.info("Inferring evaluator from raw predictions")
    item_evaluator = infer_structured_evaluator_from_predictions(
        eval_batch.target[0], text_evaluator="llm"
    )
    assert isinstance(item_evaluator, DictEval)
    return item_evaluator


def infer_from_schema() -> DictEval:
    logger.info("Inferring evaluator from schema")
    with open(SCHEMA_FILE, "r") as f:
        schema = yaml.safe_load(f)
    item_evaluator = infer_structured_evaluator_from_schema(schema, text_evaluator="llm")
    assert isinstance(item_evaluator, DictEval)
    return item_evaluator


item_evaluator = infer_from_schema()
pprint(item_evaluator)
evaluator = BatchDictEval.from_dict_eval(item_evaluator, verbose=True)

breakpoint()
logger.info("Evaluating")
results = evaluator(pred=eval_batch.pred, target=eval_batch.target)
report = EvaluationReport.from_batch_dict_eval_output(results, aggregation=AverageAggregation())

Path(SAVE_FILE).parent.mkdir(parents=True, exist_ok=True)
logger.info(f"Saving results to {SAVE_FILE}")
with open(SAVE_FILE, "w") as f:
    json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)

logger.info("Done")
