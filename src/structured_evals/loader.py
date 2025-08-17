import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from langchain_core.utils.json import parse_json_markdown
from pydantic import BaseModel

from structured_evals.parsing import parse_yaml


class EvaluationBatch(BaseModel):
    pred: list[dict[str, Any]]
    target: list[dict[str, Any]]

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        record_format: Literal["json", "yaml", None],
        pred_key: str = "pred",
        target_key: str = "target",
    ) -> "EvaluationBatch":
        data = load_results_file(path)
        parser: Callable[[Any], Any]
        if record_format == "yaml":
            parser = parse_yaml
        elif record_format == "json":
            parser = parse_json
        elif record_format is None:
            parser = identity
        else:
            raise ValueError(f"Unsupported format: {record_format}")

        preds = [parser(item[pred_key]) for item in data]
        targets = [parser(item[target_key]) for item in data]
        return cls(pred=preds, target=targets)


def parse_json(text: str) -> dict[str, Any]:
    """Parses JSON, trying parse dates as isoformat."""
    json_dict = parse_json_markdown(text)

    for key, value in json_dict.items():
        try:
            json_dict[key] = datetime.fromisoformat(value).date()
        except (ValueError, TypeError):
            pass

    return json_dict


def load_results_file(path: str | Path) -> list[dict[str, Any]]:
    """Loads results file, supporting jsonl and json."""
    path = Path(path)
    data: list[dict[str, Any]]
    if path.suffix == ".jsonl":
        data = load_jsonl(path)
    elif path.suffix == ".json":
        data = load_json(path)  # type: ignore
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    assert isinstance(data, list)
    assert all(isinstance(item, dict) for item in data)
    return data


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.loads(f.read())


def identity(x: Any) -> Any:
    return x
