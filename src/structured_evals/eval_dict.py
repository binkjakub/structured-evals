from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel
from tabulate import tabulate

from structured_evals.base import EvaluatorBase, ItemEvalOutput


class DictEvalOutput(BaseModel):
    results: dict[str, ItemEvalOutput]
    missing: dict[str, float]
    extra: dict[str, float]

    @classmethod
    def sum(cls, outputs: list["DictEvalOutput"]) -> dict[str, dict[str, float]]:
        results: dict[str, Any] = defaultdict(float)
        missing: dict[str, Any] = defaultdict(float)
        extra: dict[str, Any] = defaultdict(float)

        for output in outputs:
            for result_key, result_value in output.results.items():
                results[result_key] += result_value.score

            for missing_key, missing_value in output.missing.items():
                missing[missing_key] += missing_value

            for extra_key, extra_value in output.extra.items():
                extra[extra_key] += extra_value

        return dict(results=dict(results), missing=dict(missing), extra=dict(extra))


class DictEval(EvaluatorBase[dict[str, Any], DictEvalOutput]):
    def __init__(
        self,
        eval_mapping: dict[str, EvaluatorBase],
        error_strategy: Literal["raise", "ignore"] = "raise",
    ) -> None:
        super().__init__()
        self.eval_mapping = eval_mapping
        self.error_strategy = error_strategy

    @property
    def zero_score(self) -> DictEvalOutput:
        return DictEvalOutput(
            results={key: self.eval_mapping[key].zero_score for key in self.eval_mapping.keys()},
            missing={},
            extra={},
        )

    @property
    def max_score(self) -> DictEvalOutput:
        return DictEvalOutput(
            results={key: self.eval_mapping[key].max_score for key in self.eval_mapping.keys()},
            missing={},
            extra={},
        )

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
                results[key] = ItemEvalOutput(score=0.0)
            else:
                try:
                    results[key] = evaluator(pred[key], target[key])
                except TypeError as err:
                    if self.error_strategy == "raise":
                        raise
                    elif self.error_strategy == "ignore":
                        results[key] = ItemEvalOutput(score=0.0)
                    else:
                        raise ValueError(
                            f"Unsupported error strategy: {self.error_strategy}"
                        ) from err
        for key in pred:
            if key not in target:
                extra[key] += 1

        return DictEvalOutput(results=results, missing=dict(missing), extra=dict(extra))

    def check_dtype(self, pred: dict[str, Any], target: dict[str, Any]) -> bool:
        return isinstance(pred, dict) and isinstance(target, dict)

    def __repr__(self) -> str:
        table = []
        for key, evaluator in self.eval_mapping.items():
            table.append([key, repr(evaluator)])
        table_str = tabulate(
            table,
            headers=["Key", "Evaluator"],
            tablefmt="grid",
            maxcolwidths=[None, 40],
        )
        return f"DictEval(error_strategy={self.error_strategy})\n{table_str}"
