from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, computed_field
from tabulate import tabulate
from tqdm import tqdm

from structured_evals.base import EvaluatorBase, ItemEvalOutput
from structured_evals.eval_dict import DictEval, DictEvalOutput


class BatchDictEvalOutput(BaseModel):
    schema_keys: list[str]
    item_results: list[DictEvalOutput]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_items(self) -> int:
        return len(self.item_results)

    @property
    def scores(self) -> dict[str, list[float]]:
        scores: dict[str, list[float]] = {key: [] for key in self.schema_keys}
        for item_res in self.item_results:
            for key, score in item_res.results.items():
                scores[key].append(score.score)
        return scores

    @property
    def missing_keys(self) -> dict[str, list[float]]:
        missing_keys: dict[str, list[float]] = {key: [] for key in self.schema_keys}
        for item_res in self.item_results:
            for key, missing_key in item_res.missing_keys.items():
                missing_keys[key].append(missing_key)
        return missing_keys

    @property
    def num_times_extra_keys(self) -> dict[str, int]:
        extra_keys: dict[str, int] = defaultdict(lambda: 0)
        for item_res in self.item_results:
            for key in item_res.extra_keys.keys():
                extra_keys[key] += 1
        return extra_keys


class BatchDictEval(EvaluatorBase[list[dict[str, Any]], BatchDictEvalOutput]):
    def __init__(
        self,
        eval_mapping: dict[str, EvaluatorBase],
        error_strategy: Literal["raise", "ignore"] = "raise",
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.eval_mapping = eval_mapping
        self.schema_keys = list(eval_mapping.keys())
        self.error_strategy = error_strategy
        self.verbose = verbose

    @property
    def zero_score(self) -> BatchDictEvalOutput:
        return BatchDictEvalOutput(schema_keys=self.schema_keys, item_results=[])

    @property
    def max_score(self) -> BatchDictEvalOutput:
        return BatchDictEvalOutput(schema_keys=self.schema_keys, item_results=[])

    def evaluate(
        self,
        pred: list[dict[str, Any]],
        target: list[dict[str, Any]],
    ) -> BatchDictEvalOutput:
        # TODO: handle cases when pred wasn't parsed

        num_items = len(target)
        schema_keys = list(self.eval_mapping.keys())

        for target_item in target:
            unspecified_keys = {key for key in target_item if key not in self.eval_mapping}
            if unspecified_keys:
                raise ValueError(
                    f"Target dict contains keys not present in eval_mapping: {unspecified_keys}"
                )

        valid_results = {}
        missing_results: dict[str, list[ItemEvalOutput]] = {key: [] for key in schema_keys}
        missing_mask: dict[str, list[int]] = {key: [] for key in schema_keys}

        with tqdm(
            self.eval_mapping.items(),
            disable=not self.verbose,
        ) as pbar:
            for key, evaluator in pbar:
                pbar.set_description(f"Evaluating key: {key} ({evaluator.name})")
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
                    missing_keys={key: missing_mask[key][i] for key in schema_keys},
                    extra_keys={key: 1 for key in pred[i].keys() if key not in schema_keys},
                )
            )

        return BatchDictEvalOutput(
            schema_keys=schema_keys,
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

    @classmethod
    def from_dict_eval(cls, dict_eval: DictEval, verbose: bool) -> "BatchDictEval":
        return cls(
            eval_mapping=dict_eval.eval_mapping,
            error_strategy=dict_eval.error_strategy,
            verbose=verbose,
        )
