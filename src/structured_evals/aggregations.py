from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np

from structured_evals import BatchDictEvalOutput


def get_aggregation(aggregation: str) -> "Aggregation":
    if aggregation == "average":
        return AverageAggregation()
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")


class Aggregation(ABC):
    @abstractmethod
    def __call__(self, outs: BatchDictEvalOutput) -> dict[str, Any]:
        raise NotImplementedError("Aggregation subclasses must implement __call__")


class AverageAggregation(Aggregation):
    def __call__(self, outs: BatchDictEvalOutput) -> dict[str, Any]:
        mean: dict[str, Any] = {}
        standard_error: dict[str, Any] = {}
        mean_times_missing: dict[str, Any] = {}
        mean_times_extra: dict[str, Any] = {}

        for key in outs.scores:
            mean[key] = float(np.mean(outs.scores[key]))
            standard_error[key] = float(np.std(outs.scores[key]) / np.sqrt(outs.num_items))

        for key in outs.missing_keys:
            if outs.missing_keys[key]:
                mean_times_missing[key] = float(np.mean(outs.missing_keys[key]))
            else:
                mean_times_missing[key] = 0.0

        for key in outs.num_times_extra_keys:
            mean_times_extra[key] = outs.num_times_extra_keys[key] / outs.num_items

        return {
            "mean": mean,
            "standard_error": standard_error,
            "mean_times_missing": mean_times_missing,
            "mean_times_extra": mean_times_extra,
        }


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

    def __call__(self, outs: BatchDictEvalOutput) -> dict[str, Any]:
        # TODO: refactor with new signature of BatchDictEvalOutput
        relevant_retrieved: list[float] = []
        all_retrieved: list[int] = []
        all_relevant: list[int] = []

        for out in outs.item_results:
            all_retrieved.append(len(out.results) + len(out.extra_keys) - len(out.missing_keys))
            all_relevant.append(len(out.results.values()))

            if self.mode == "hard":
                relevant_retrieved.append(sum(float(val.score > 0) for val in out.results.values()))
            elif self.mode == "soft":
                relevant_retrieved.append(sum(val.score for val in out.results.values()))
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
