from pprint import pprint

from structured_evals import EvaluationBatch, infer_structured_evaluator
from structured_evals.eval_batch import BatchDictEval
from structured_evals.eval_dict import DictEval

eval_batch = EvaluationBatch.from_json(
    path="data/example_real_schema.json",
    record_format="yaml",
    pred_key="answer",
    target_key="gold",
)

item_evaluator = infer_structured_evaluator(eval_batch.target[0])
assert isinstance(item_evaluator, DictEval)
evaluator = BatchDictEval(
    eval_mapping=item_evaluator.eval_mapping,
    aggregation="average",
    error_strategy="raise",
)

results = evaluator(pred=eval_batch.pred, target=eval_batch.target)

pprint(results.agg_results)
