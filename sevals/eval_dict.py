from dataclasses import dataclass
from typing import Any

from sevals.base import Evaluator


@dataclass(kw_only=True)
class DictEvalOutput:
    results: dict[str, float]
    missing: list[str]
    extra: list[str]


class DictEval(Evaluator[dict[str, Any], DictEvalOutput]):
    def __init__(self, eval_mapping: dict[str, Evaluator]) -> None:
        super().__init__()
        self.eval_mapping = eval_mapping

    def evaluate(self, pred: dict[str, Any], target: dict[str, Any]) -> DictEvalOutput:
        results = {}
        missing = []
        extra = [key for key in pred if key not in self.eval_mapping]

        if any(key not in self.eval_mapping for key in target):
            raise ValueError(
                "Target dict contains keys not present in eval_mapping, you must provide a target coherent with eval_mapping"
            )

        for key, eval_ in self.eval_mapping.items():
            if key not in pred:
                missing.append(key)
                results[key] = 0.0
            else:
                results[key] = eval_.evaluate(pred[key], target[key])

        return DictEvalOutput(results=results, missing=missing, extra=extra)
