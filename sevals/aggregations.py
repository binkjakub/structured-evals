from abc import ABC, abstractmethod
from typing import Any

from sevals.eval_dict import DictEvalOutput


def get_aggregation(aggregation: str) -> "Aggregation":
    if aggregation == "average":
        return AverageAggregation()
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")


class Aggregation(ABC):
    @abstractmethod
    def __call__(self, outs: list[DictEvalOutput]) -> dict[str, Any]:
        raise NotImplementedError("Aggregation subclasses must implement __call__")


class AverageAggregation(Aggregation):
    def __call__(self, outs: list[DictEvalOutput]) -> dict[str, Any]:
        outs_sum = DictEvalOutput.sum(outs)

        results: dict[str, Any] = {}
        missing: dict[str, Any] = {}
        extra: dict[str, Any] = {}

        for key in outs_sum.results:
            results[key] /= len(outs)

        for key in outs_sum.missing:
            missing[key] /= len(outs)

        for key in outs_sum.extra:
            extra[key] /= len(outs)

        return {"results": results, "missing": missing, "extra": extra}


class F1ScoreAggregation(Aggregation):
    def __call__(self, outs: list[DictEvalOutput]) -> dict[str, Any]:
        raise NotImplementedError()
