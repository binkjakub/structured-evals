from pprint import pprint

from structured_evals import EvaluationBatch, infer_structured_evaluator

eval_batch = EvaluationBatch.from_json(
    "data/example_real_schema.jsonl",
    record_format="json",
    pred_key="answer",
    target_key="gold",
)


evaluator = infer_structured_evaluator(eval_batch.target)
pprint(evaluator)
