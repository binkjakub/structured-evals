import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sevals.parsing import parse_yaml


@dataclass(frozen=True, kw_only=True, slots=True)
class EvaluationBatch:
    pred: list[dict[str, Any]]
    target: list[dict[str, Any]]

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        pred_key: str = "pred",
        target_key: str = "target",
    ) -> "EvaluationBatch":
        data = load_json(path)
        return cls(pred=data[pred_key], target=data[target_key])

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        pred_key: str = "pred",
        target_key: str = "target",
    ) -> "EvaluationBatch":
        preds: list[dict[str, Any]] = []
        targets: list[dict[str, Any]] = []
        for item in load_jsonl(path):
            preds.append(parse_yaml(item[pred_key]))
            targets.append(parse_yaml(item[target_key]))
        return cls(pred=preds, target=targets)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        pred_key: str = "pred",
        target_key: str = "target",
    ) -> "EvaluationBatch":
        raise NotImplementedError()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)
