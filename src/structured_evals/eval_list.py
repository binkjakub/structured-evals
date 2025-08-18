from typing import Any, Literal

import numpy as np

from structured_evals.base import EvaluatorBase, ItemEvalOutput

T_list_aggregation = Literal["average", "sum"]


class ListEvalOutput(ItemEvalOutput):
    num_missing_items: int
    num_extra_items: int


class ListEval(EvaluatorBase[list[Any], ListEvalOutput]):
    def __init__(
        self,
        item_evaluator: EvaluatorBase[Any, ItemEvalOutput],
        aggregation: T_list_aggregation = "average",
    ) -> None:
        super().__init__()
        self.item_evaluator = item_evaluator
        self.aggregation = aggregation

    @property
    def zero_score(self) -> ListEvalOutput:
        return ListEvalOutput(score=0.0, num_missing_items=0, num_extra_items=0)

    @property
    def max_score(self) -> ListEvalOutput:
        return ListEvalOutput(score=1.0, num_missing_items=0, num_extra_items=0)

    def evaluate(self, pred: list[Any], target: list[Any]) -> ListEvalOutput:
        if self.is_null(pred) and self.is_null(target):
            return self.max_score
        elif self.is_null(pred) or self.is_null(target):
            return self.zero_score
        elif not self.check_dtype(pred, target):
            return self.zero_score

        # TODO: implement with hungarian algorithm instead of greedy matching
        target_pred_similarity = []

        for target_item in target:
            pred_similarities = []
            for pred_item in pred:
                pred_similarities.append(self.item_evaluator(pred_item, target_item).score)
            target_pred_similarity.append(pred_similarities)

        sim = np.array(target_pred_similarity, dtype=float)
        preds_queue = list(range(sim.shape[1]))
        results = {
            "score": 0,
            "missing": 0,
            "extra": 0,
        }
        for i in range(sim.shape[0]):
            if not preds_queue:
                results["missing"] += 1
            else:
                best_pred_idx = np.argmax(sim[i][preds_queue])
                best_pred_score = sim[i][preds_queue][best_pred_idx]
                results["score"] += best_pred_score
                preds_queue.pop(best_pred_idx)

        results["score"] /= sim.shape[0]
        results["extra"] += len(preds_queue)

        return ListEvalOutput(**results)

    def is_null(self, item: Any) -> bool:
        return item is None or item == "" or item == []

    def check_dtype(self, pred: list[Any], target: list[Any]) -> bool:
        return isinstance(pred, list) and isinstance(target, list)
