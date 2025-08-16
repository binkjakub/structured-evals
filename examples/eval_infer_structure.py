from pprint import pprint

from sevals import EvaluationBatch, infer_structured_evaluator

eval_batch = EvaluationBatch.from_json(
    "data/example_real_schema.jsonl", pred_key="answer", target_key="gold"
)


evaluator = infer_structured_evaluator(eval_batch.target)
pprint(evaluator)

# results = evaluator(pred=eval_batch.pred, target=eval_batch.target)

# pprint(results.agg_results)
