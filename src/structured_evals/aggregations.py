from abc import ABC, abstractmethod
from typing import Any, Literal

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

        for key in outs_sum["results"]:
            results[key] = outs_sum["results"][key] / len(outs)

        for key in outs_sum["missing"]:
            missing[key] = outs_sum["missing"][key] / len(outs)

        for key in outs_sum["extra"]:
            extra[key] = outs_sum["extra"][key] / len(outs)

        return {"results": results, "missing": missing, "extra": extra}


class F1ScoreAggregation(Aggregation):
    """Aggregates F1 score, precision, and recall for multiple evaluations.
    - Precision: measures the proportion of relevant keys extracted by a model among all the extracted items.
    - Recall: measures the proportion of relevant keys extracted by a model among all the relevant items.
    - F1 score: the harmonic mean of precision and recall.

    Operates in two modes:
        - Hard: treats each score as 1 if the score (e.g. ROUGE, BLEU,...) is greater than 0.
        - Soft: aggregates its scores directly.

    Operates in two averages:
        - Micro: computes the average after summing the scores over all the evaluations.
        - Macro: computes the average of the precision, recall, f1 over all the evaluations.
    """

    def __init__(
        self,
        mode: Literal["hard", "soft"],
        average: Literal["micro", "macro"] = "micro",
    ) -> None:
        self.mode = mode
        self.average = average

    def __call__(self, outs: list[DictEvalOutput]) -> dict[str, Any]:
        relevant_retrieved: list[float] = []
        all_retrieved: list[int] = []
        all_relevant: list[int] = []

        for out in outs:
            all_retrieved.append(len(out.results) + len(out.extra) - len(out.missing))
            all_relevant.append(len(out.results.values()))

            if self.mode == "hard":
                relevant_retrieved.append(sum(float(val > 0) for val in out.results.values()))
            elif self.mode == "soft":
                relevant_retrieved.append(sum(out.results.values()))
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

        if self.average == "micro":
            return self._micro_average(relevant_retrieved, all_retrieved, all_relevant)
        elif self.average == "macro":
            return self._macro_average(relevant_retrieved, all_retrieved, all_relevant)
        else:
            raise ValueError(f"Unsupported average: {self.average}")

    @staticmethod
    def _micro_average(
        relevant_retrieved: list[float],
        all_retrieved: list[int],
        all_relevant: list[int],
    ) -> dict[str, float]:
        precision = sum(relevant_retrieved) / sum(all_retrieved)
        recall = sum(relevant_retrieved) / sum(all_relevant)
        f1 = 2 * precision * recall / (precision + recall)
        return {"f1": f1, "precision": precision, "recall": recall}

    @staticmethod
    def _macro_average(
        relevant_retrieved: list[float],
        all_retrieved: list[int],
        all_relevant: list[int],
    ) -> dict[str, float]:
        num_items = len(relevant_retrieved)
        precisions = [
            relevant_retrieved[i] / all_retrieved[i] if all_retrieved[i] != 0 else 0
            for i in range(num_items)
        ]
        recalls = [
            relevant_retrieved[i] / all_relevant[i] if all_relevant[i] != 0 else 0
            for i in range(num_items)
        ]
        f1_scores = [
            (
                2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i])
                if precisions[i] + recalls[i] != 0
                else 0
            )
            for i in range(num_items)
        ]
        precision = sum(precisions) / num_items
        recall = sum(recalls) / num_items
        f1 = sum(f1_scores) / num_items
        return {"f1": f1, "precision": precision, "recall": recall}
