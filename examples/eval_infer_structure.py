import json
from pprint import pprint

from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from structured_evals import EvaluationBatch, infer_structured_evaluator
from structured_evals.aggregations import AverageAggregation
from structured_evals.eval_batch import BatchDictEval
from structured_evals.eval_dict import DictEval
from structured_evals.report import EvaluationReport

set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))

eval_batch = EvaluationBatch.from_json(
    path="data/example_real_schema.json",
    record_format="yaml",
    pred_key="answer",
    target_key="gold",
)

item_evaluator = infer_structured_evaluator(eval_batch.target[0], text_evaluator="llm")
assert isinstance(item_evaluator, DictEval)
evaluator = BatchDictEval(
    eval_mapping=item_evaluator.eval_mapping,
    error_strategy="raise",
    verbose=True,
)
pprint(item_evaluator)

results = evaluator(pred=eval_batch.pred, target=eval_batch.target)
report = EvaluationReport.from_batch_dict_eval_output(results, aggregation=AverageAggregation())
with open(".local/results.json", "w") as f:
    json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
