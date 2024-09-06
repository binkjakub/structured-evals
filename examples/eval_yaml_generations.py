from pprint import pprint

from torchmetrics.functional.text import chrf_score

from sevals.eval_batch import BatchDictEval
from sevals.eval_dict import DictEval
from sevals.eval_primitive import DateEval, NumEval
from sevals.eval_text import EvalTextualMetric
from sevals.loader import EvaluationBatch

eval_batch = EvaluationBatch.from_jsonl("data/sample_extra_missing.jsonl")


def chrf_eval(pred: str, target: str) -> float:
    return chrf_score([pred], [target], n_char_order=1, n_word_order=0).item()  # type: ignore


evaluator = BatchDictEval(
    item_evaluator=DictEval(
        eval_mapping={
            "name": EvalTextualMetric(chrf_eval, "chrf"),
            "age": NumEval(),
            "birthday": DateEval(),
        }
    ),
    aggregation="average",
    error_strategy="ignore",
)

results = evaluator(pred=eval_batch.pred, target=eval_batch.target)

pprint(results.agg_results)
