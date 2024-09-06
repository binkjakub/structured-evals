from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from sevals.base import Evaluator


@dataclass(kw_only=True)
class DictEvalOutput:
    results: dict[str, float]
    missing: dict[str, float]
    extra: dict[str, float]

    @staticmethod
    def aggregate(
        items: list["DictEvalOutput"], aggregation: Literal["average"]
    ) -> "DictEvalOutput":
        if aggregation == "average":
            results: dict[str, float] = defaultdict(float)
            missing: dict[str, float] = defaultdict(int)
            extra: dict[str, float] = defaultdict(int)

            for item in items:
                for key, value in item.results.items():
                    results[key] += value

                for key, value in item.missing.items():
                    missing[key] += value

                for key, value in item.extra.items():
                    extra[key] += value

            for key in results:
                results[key] /= len(items)

            return DictEvalOutput(results=dict(results), missing=dict(missing), extra=dict(extra))
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")


class DictEval(Evaluator[dict[str, Any], DictEvalOutput]):
    def __init__(self, eval_mapping: dict[str, Evaluator]) -> None:
        super().__init__()
        self.eval_mapping = eval_mapping

    def evaluate(self, pred: dict[str, Any], target: dict[str, Any]) -> DictEvalOutput:
        if any(key not in self.eval_mapping for key in target):
            raise ValueError(
                "Target dict contains keys not present in eval_mapping, you must provide a target coherent with eval_mapping"
            )

        results = {}
        missing: dict[str, float] = defaultdict(float)
        extra: dict[str, float] = defaultdict(float)

        for key, eval_ in self.eval_mapping.items():
            if key not in pred:
                missing[key] += 1
                results[key] = 0.0
            else:
                results[key] = eval_.evaluate(pred[key], target[key])

        for key in pred:
            if key not in target:
                extra[key] += 1

        return DictEvalOutput(results=results, missing=dict(missing), extra=dict(extra))
