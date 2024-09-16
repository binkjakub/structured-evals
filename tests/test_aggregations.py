import pytest

from sevals.aggregations import AverageAggregation, F1ScoreAggregation
from sevals.eval_dict import DictEvalOutput


def test_average_aggregation() -> None:
    aggregation = AverageAggregation()
    outs = [
        DictEvalOutput(
            results={"a": 1, "b": 0},
            missing={"b": 1},
            extra={},
        ),
        DictEvalOutput(
            results={"a": 0.5, "b": 0},
            missing={"b": 1},
            extra={"c": 1},
        ),
    ]

    assert {
        "results": {"a": 0.75, "b": 0.0},
        "missing": {"b": 1.0},
        "extra": {"c": 0.5},
    } == aggregation(outs)


def test_f1_hard_macro_aggregation() -> None:
    aggregation = F1ScoreAggregation(mode="hard", average="macro")

    outs = [
        DictEvalOutput(
            results={"a": 0.75, "b": 0},
            missing={"b": 1},
            extra={},
        ),
        DictEvalOutput(
            results={"a": 1, "b": 0},
            missing={"b": 1},
            extra={"c": 1},
        ),
        DictEvalOutput(
            results={"a": 1, "b": 0.5},
            missing={},
            extra={"c": 1, "d": 1},
        ),
    ]

    assert pytest.approx(
        {
            "precision": 2 / 3,
            "recall": 2 / 3,
            "f1": 0.6111111,
        }
    ) == aggregation(outs)


def test_f1_hard_micro_aggregation() -> None:
    aggregation = F1ScoreAggregation(mode="hard", average="micro")

    outs = [
        DictEvalOutput(
            results={"a": 0.75, "b": 0},
            missing={"b": 1},
            extra={},
        ),
        DictEvalOutput(
            results={"a": 1, "b": 0},
            missing={"b": 1},
            extra={"c": 1},
        ),
        DictEvalOutput(
            results={"a": 1, "b": 0.5},
            missing={},
            extra={"c": 1, "d": 1},
        ),
    ]

    assert pytest.approx(
        {
            "precision": 4 / 7,
            "recall": 4 / 6,
            "f1": 0.6153846153846153,
        }
    ) == aggregation(outs)


def test_f1_soft_micro_aggregation() -> None:
    aggregation = F1ScoreAggregation(mode="soft", average="micro")

    outs = [
        DictEvalOutput(
            results={"a": 0.75, "b": 0},
            missing={"b": 1},
            extra={},
        ),
        DictEvalOutput(
            results={"a": 1, "b": 0},
            missing={"b": 1},
            extra={"c": 1},
        ),
        DictEvalOutput(
            results={"a": 0.25, "b": 0.5},
            missing={},
            extra={"c": 1, "d": 1},
        ),
    ]

    assert pytest.approx(
        {
            "precision": 2.5 / 7,
            "recall": 2.5 / 6,
            "f1": 0.3846153846153846,
        }
    ) == aggregation(outs)


def test_f1_soft_macro_aggregation() -> None:
    aggregation = F1ScoreAggregation(mode="soft", average="macro")

    outs = [
        DictEvalOutput(
            results={"a": 0.75, "b": 0},
            missing={"b": 1},
            extra={},
        ),
        DictEvalOutput(
            results={"a": 1, "b": 0},
            missing={"b": 1},
            extra={"c": 1},
        ),
        DictEvalOutput(
            results={"a": 0.25, "b": 0.5},
            missing={},
            extra={"c": 1, "d": 1},
        ),
    ]

    assert pytest.approx(
        {
            "precision": (0.75 + 0.5 + 0.75 / 4) / 3,
            "recall": (0.75 / 2 + 0.5 + 0.75 / 2) / 3,
            "f1": (0.5 + 0.5 + 0.25) / 3,
        }
    ) == aggregation(outs)
