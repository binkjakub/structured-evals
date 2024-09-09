from abc import ABC, abstractmethod
from collections import defaultdict
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
        results: dict[str, float] = defaultdict(float)
        missing: dict[str, float] = defaultdict(int)
        extra: dict[str, float] = defaultdict(int)

        for item in outs:
            for key, value in item.results.items():
                results[key] += value

            for key, value in item.missing.items():
                missing[key] += value

            for key, value in item.extra.items():
                extra[key] += value

        for key in results:
            results[key] /= len(outs)

        for key in missing:
            missing[key] /= len(outs)

        for key in extra:
            extra[key] /= len(outs)

        return dict(results=dict(results), missing=dict(missing), extra=dict(extra))
