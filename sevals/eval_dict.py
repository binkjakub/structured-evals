from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from sevals.base import EvaluatorBase


@dataclass(kw_only=True, frozen=True)
class DictEvalOutput:
    results: dict[str, float]
    missing: dict[str, float]
    extra: dict[str, float]

    @classmethod
    def sum(cls, outputs: list["DictEvalOutput"]) -> "DictEvalOutput":
        results: dict[str, Any] = defaultdict(float)
        missing: dict[str, Any] = defaultdict(float)
        extra: dict[str, Any] = defaultdict(float)

        for output in outputs:
            for key, value in output.results.items():
                results[key] += value

            for key, value in output.missing.items():
                missing[key] += value

            for key, value in output.extra.items():
                extra[key] += value

        return DictEvalOutput(results=dict(results), missing=dict(missing), extra=dict(extra))


class DictEval(EvaluatorBase[dict[str, Any], DictEvalOutput]):
    def __init__(
        self,
        eval_mapping: dict[str, EvaluatorBase],
        error_strategy: Literal["raise", "ignore"] = "raise",
    ) -> None:
        super().__init__()
        self.eval_mapping = eval_mapping
        self.error_strategy = error_strategy

    def evaluate(self, pred: dict[str, Any], target: dict[str, Any]) -> DictEvalOutput:
        if any(key not in self.eval_mapping for key in target):
            raise ValueError(
                "Target dict contains keys not present in eval_mapping, you must provide a target coherent with eval_mapping"
            )

        results = {}
        missing: dict[str, float] = defaultdict(float)
        extra: dict[str, float] = defaultdict(float)

        for key, evaluator in self.eval_mapping.items():
            if key not in pred:
                missing[key] += 1
                results[key] = 0.0
            else:
                try:
                    results[key] = evaluator(pred[key], target[key])
                except TypeError as err:
                    if self.error_strategy == "raise":
                        raise
                    elif self.error_strategy == "ignore":
                        results[key] = 0.0
                    else:
                        raise ValueError(
                            f"Unsupported error strategy: {self.error_strategy}"
                        ) from err

        for key in pred:
            if key not in target:
                extra[key] += 1

        return DictEvalOutput(results=results, missing=dict(missing), extra=dict(extra))

    def check_dtype(self, pred: dict[str, Any], target: dict[str, Any]) -> None:
        if not isinstance(pred, dict) or not isinstance(target, dict):
            raise ValueError("Both pred and target must be dictionaries: dict[str, Any].")
