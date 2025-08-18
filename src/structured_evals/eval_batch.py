from typing import Any, Literal

from pydantic import BaseModel, computed_field
from tabulate import tabulate

from structured_evals.aggregations import get_aggregation
from structured_evals.base import EvaluatorBase, ItemEvalOutput
from structured_evals.eval_dict import DictEvalOutput


class BatchDictEvalOutput(BaseModel):
    agg_results: dict[str, Any]
    item_results: list[DictEvalOutput]

    @property
    @computed_field
    def total_items(self) -> int:
        return len(self.item_results)


class BatchDictEval(EvaluatorBase[list[dict[str, Any]], BatchDictEvalOutput]):
    def __init__(
        self,
        eval_mapping: dict[str, EvaluatorBase],
        aggregation: str,
        error_strategy: Literal["raise", "ignore"] = "raise",
    ) -> None:
        super().__init__()
        self.eval_mapping = eval_mapping
        self.error_strategy = error_strategy
        self.aggregation = get_aggregation(aggregation)

    @property
    def zero_score(self) -> BatchDictEvalOutput:
        return BatchDictEvalOutput(agg_results={}, item_results=[])

    @property
    def max_score(self) -> BatchDictEvalOutput:
        return BatchDictEvalOutput(agg_results={}, item_results=[])

    def evaluate(
        self,
        pred: list[dict[str, Any]],
        target: list[dict[str, Any]],
    ) -> BatchDictEvalOutput:
        # TODO: handle cases when pred wasn't parsed

        num_items = len(target)
        schema_keys = list(self.eval_mapping.keys())

        for target_item in target:
            if any(key not in self.eval_mapping for key in target_item):
                raise ValueError(
                    "Target dict contains keys not present in eval_mapping, you must provide a target coherent with eval_mapping"
                )

        valid_results = {}
        missing_results: dict[str, list[ItemEvalOutput]] = {key: [] for key in schema_keys}
        missing_mask: dict[str, list[int]] = {key: [] for key in schema_keys}

        for key, evaluator in self.eval_mapping.items():
            eval_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for pred_item, target_item in zip(pred, target, strict=True):
                if key not in pred_item:
                    missing_mask[key].append(1)
                    missing_results[key].append(evaluator.zero_score)
                else:
                    missing_mask[key].append(0)
                    eval_pairs.append((pred_item[key], target_item[key]))

            if hasattr(evaluator, "evaluate_batch"):
                valid_results[key] = evaluator.evaluate_batch(*zip(*eval_pairs))
            else:
                valid_results[key] = [evaluator.evaluate(*pair) for pair in eval_pairs]

        assert all(
            len(missing_results[key]) + len(valid_results[key]) == num_items for key in schema_keys
        )

        results = []
        valid_res_idx: dict[str, int] = dict.fromkeys(schema_keys, 0)
        missing_res_idx: dict[str, int] = dict.fromkeys(schema_keys, 0)
        for i in range(num_items):
            item_res = {}

            for key in schema_keys:
                if missing_mask[key][i]:
                    item_res[key] = missing_results[key][missing_res_idx[key]]
                    missing_res_idx[key] += 1
                else:
                    item_res[key] = valid_results[key][valid_res_idx[key]]
                    valid_res_idx[key] += 1

            results.append(
                DictEvalOutput(
                    results=item_res,
                    missing={key: missing_mask[key][i] for key in schema_keys},
                    extra={key: 1 for key in pred[i].keys() if key not in schema_keys},
                )
            )

        return BatchDictEvalOutput(
            agg_results=self.aggregation(results),
            item_results=results,
        )

    def check_dtype(self, pred: list[dict[str, Any]], target: list[dict[str, Any]]) -> bool:
        return isinstance(pred, list) and isinstance(target, list) and len(pred) == len(target)

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
