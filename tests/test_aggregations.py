import pytest

from structured_evals.aggregations import AverageAggregation, F1ScoreAggregation
from structured_evals.base import ItemEvalOutput
from structured_evals.eval_batch import BatchDictEvalOutput
from structured_evals.eval_dict import DictEvalOutput


def test_average_aggregation() -> None:
    aggregation = AverageAggregation()
    outs = BatchDictEvalOutput(
        schema_keys=["a", "b"],
        item_results=[
            DictEvalOutput(
                results={"a": ItemEvalOutput(score=1), "b": ItemEvalOutput(score=0)},
                missing_keys={"b": 1},
                extra_keys={},
            ),
            DictEvalOutput(
                results={"a": ItemEvalOutput(score=0.5), "b": ItemEvalOutput(score=0)},
                missing_keys={"b": 1},
                extra_keys={"c": 1},
            ),
        ],
    )

    assert {
        "mean": {"a": 0.75, "b": 0.0},
        "standard_error": {"a": pytest.approx(0.17677, abs=1e-3), "b": 0.0},
        "mean_times_missing": {"a": 0.0, "b": 1.0},
        "mean_times_extra": {"c": 0.5},
    } == aggregation(outs)


def test_f1_hard_macro_aggregation() -> None:
    aggregation = F1ScoreAggregation(mode="hard", average="macro")

    outs = BatchDictEvalOutput(
        schema_keys=["a", "b"],
        item_results=[
            DictEvalOutput(
                results={"a": ItemEvalOutput(score=0.75), "b": ItemEvalOutput(score=0)},
                missing_keys={"b": 1},
                extra_keys={},
            ),
            DictEvalOutput(
                results={"a": ItemEvalOutput(score=1), "b": ItemEvalOutput(score=0)},
                missing_keys={"b": 1},
                extra_keys={"c": 1},
            ),
            DictEvalOutput(
                results={"a": ItemEvalOutput(score=1), "b": ItemEvalOutput(score=0.5)},
                missing_keys={},
                extra_keys={"c": 1, "d": 1},
            ),
        ],
    )

    assert pytest.approx(
        {
            "precision": 2 / 3,
            "recall": 2 / 3,
            "f1": 0.6111111,
        }
    ) == aggregation(outs)


def test_f1_hard_micro_aggregation() -> None:
    aggregation = F1ScoreAggregation(mode="hard", average="micro")

    outs = BatchDictEvalOutput(
        schema_keys=["a", "b"],
        item_results=[
            DictEvalOutput(
                results={"a": ItemEvalOutput(score=0.75), "b": ItemEvalOutput(score=0)},
                missing_keys={"b": 1},
                extra_keys={},
            ),
            DictEvalOutput(
                results={"a": ItemEvalOutput(score=1), "b": ItemEvalOutput(score=0)},
                missing_keys={"b": 1},
                extra_keys={"c": 1},
            ),
            DictEvalOutput(
                results={"a": ItemEvalOutput(score=1), "b": ItemEvalOutput(score=0.5)},
                missing_keys={},
                extra_keys={"c": 1, "d": 1},
            ),
        ],
    )

    assert pytest.approx(
        {
            "precision": 4 / 7,
            "recall": 4 / 6,
            "f1": 0.6153846153846153,
        }
    ) == aggregation(outs)


def test_f1_soft_micro_aggregation() -> None:
    aggregation = F1ScoreAggregation(mode="soft", average="micro")

    outs = BatchDictEvalOutput(
        schema_keys=["a", "b"],
        item_results=[
            DictEvalOutput(
                results={"a": ItemEvalOutput(score=0.75), "b": ItemEvalOutput(score=0)},
                missing_keys={"b": 1},
                extra_keys={},
            ),
            DictEvalOutput(
                results={"a": ItemEvalOutput(score=1), "b": ItemEvalOutput(score=0)},
                missing_keys={"b": 1},
                extra_keys={"c": 1},
            ),
            DictEvalOutput(
                results={"a": ItemEvalOutput(score=0.25), "b": ItemEvalOutput(score=0.5)},
                missing_keys={},
                extra_keys={"c": 1, "d": 1},
            ),
        ],
    )

    assert pytest.approx(
        {
            "precision": 2.5 / 7,
            "recall": 2.5 / 6,
            "f1": 0.3846153846153846,
        }
    ) == aggregation(outs)


def test_f1_soft_macro_aggregation() -> None:
    aggregation = F1ScoreAggregation(mode="soft", average="macro")

    outs = BatchDictEvalOutput(
        schema_keys=["a", "b"],
        item_results=[
            DictEvalOutput(
                results={"a": ItemEvalOutput(score=0.75), "b": ItemEvalOutput(score=0)},
                missing_keys={"b": 1},
                extra_keys={},
            ),
            DictEvalOutput(
                results={"a": ItemEvalOutput(score=1), "b": ItemEvalOutput(score=0)},
                missing_keys={"b": 1},
                extra_keys={"c": 1},
            ),
            DictEvalOutput(
                results={"a": ItemEvalOutput(score=0.25), "b": ItemEvalOutput(score=0.5)},
                missing_keys={},
                extra_keys={"c": 1, "d": 1},
            ),
        ],
    )

    assert pytest.approx(
        {
            "precision": (0.75 + 0.5 + 0.75 / 4) / 3,
            "recall": (0.75 / 2 + 0.5 + 0.75 / 2) / 3,
            "f1": (0.5 + 0.5 + 0.25) / 3,
        }
    ) == aggregation(outs)
