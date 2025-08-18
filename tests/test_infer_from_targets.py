from datetime import datetime
from typing import Any

import pytest

from structured_evals.eval_dict import DictEval
from structured_evals.eval_list import ListEval
from structured_evals.eval_primitive import DateEval, NumEval
from structured_evals.eval_text import EvalTextualMetric
from structured_evals.infer_from_targets import (
    DEFAULT_LIST_AGGREGATION,
    infer_structured_evaluator_from_predictions,
)


def test_infer_evaluator_for_string() -> None:
    """Test that string data returns EvalTextualMetric evaluator."""
    data = "test string"
    evaluator = infer_structured_evaluator_from_predictions(data, text_evaluator="ngram")

    assert isinstance(evaluator, EvalTextualMetric)
    assert evaluator.name == "chrf"


def test_infer_evaluator_for_integer() -> None:
    """Test that integer data returns NumEval evaluator."""
    data = 42
    evaluator = infer_structured_evaluator_from_predictions(data, text_evaluator="ngram")

    assert isinstance(evaluator, NumEval)


def test_infer_evaluator_for_float() -> None:
    """Test that float data returns NumEval evaluator."""
    data = 3.14
    evaluator = infer_structured_evaluator_from_predictions(data, text_evaluator="ngram")

    assert isinstance(evaluator, NumEval)


def test_infer_evaluator_for_datetime() -> None:
    """Test that datetime data returns DateEval evaluator."""
    data = datetime(2021, 1, 1)
    evaluator = infer_structured_evaluator_from_predictions(data, text_evaluator="ngram")

    assert isinstance(evaluator, DateEval)


def test_infer_evaluator_for_simple_dict() -> None:
    """Test that dict data returns DictEval evaluator with correct mapping."""
    data = {"name": "John", "age": 30}
    evaluator = infer_structured_evaluator_from_predictions(data, text_evaluator="ngram")

    assert isinstance(evaluator, DictEval)
    assert "name" in evaluator.eval_mapping
    assert "age" in evaluator.eval_mapping
    assert isinstance(evaluator.eval_mapping["name"], EvalTextualMetric)
    assert isinstance(evaluator.eval_mapping["age"], NumEval)


def test_infer_evaluator_for_nested_dict() -> None:
    """Test that nested dict creates properly nested DictEval evaluators."""
    data = {"user": {"name": "John", "age": 30}, "created_at": datetime(2021, 1, 1)}
    evaluator = infer_structured_evaluator_from_predictions(data, text_evaluator="ngram")

    assert isinstance(evaluator, DictEval)
    assert isinstance(evaluator.eval_mapping["user"], DictEval)
    assert isinstance(evaluator.eval_mapping["created_at"], DateEval)

    # Check nested dict evaluators
    user_evaluator = evaluator.eval_mapping["user"]
    assert isinstance(user_evaluator.eval_mapping["name"], EvalTextualMetric)
    assert isinstance(user_evaluator.eval_mapping["age"], NumEval)


def test_infer_evaluator_for_list_of_strings() -> None:
    """Test that list of strings returns ListEval with string item evaluator."""
    data = ["apple", "banana", "cherry"]
    evaluator = infer_structured_evaluator_from_predictions(data, text_evaluator="ngram")

    assert isinstance(evaluator, ListEval)
    assert isinstance(evaluator.item_evaluator, EvalTextualMetric)
    assert evaluator.aggregation == "average"


def test_infer_evaluator_for_list_of_numbers() -> None:
    """Test that list of numbers returns ListEval with NumEval item evaluator."""
    data = [1, 2, 3]
    evaluator = infer_structured_evaluator_from_predictions(data, text_evaluator="ngram")

    assert isinstance(evaluator, ListEval)
    assert isinstance(evaluator.item_evaluator, NumEval)


def test_infer_evaluator_for_list_of_dicts() -> None:
    """Test that list of dicts returns ListEval with DictEval item evaluator."""
    data = [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]
    evaluator = infer_structured_evaluator_from_predictions(data, text_evaluator="ngram")

    assert isinstance(evaluator, ListEval)
    assert isinstance(evaluator.item_evaluator, DictEval)

    # Check the dict evaluator structure
    dict_evaluator = evaluator.item_evaluator
    assert isinstance(dict_evaluator.eval_mapping["name"], EvalTextualMetric)
    assert isinstance(dict_evaluator.eval_mapping["age"], NumEval)


def test_infer_evaluator_for_complex_nested_structure() -> None:
    """Test complex nested structure with mixed types."""
    data = {
        "users": [
            {
                "name": "John",
                "age": 30,
                "metadata": {"created_at": datetime(2021, 1, 1), "score": 95.5},
            }
        ],
        "count": 1,
        "description": "User list",
    }
    evaluator = infer_structured_evaluator_from_predictions(data, text_evaluator="ngram")

    assert isinstance(evaluator, DictEval)

    # Check top-level structure
    assert isinstance(evaluator.eval_mapping["users"], ListEval)
    assert isinstance(evaluator.eval_mapping["count"], NumEval)
    assert isinstance(evaluator.eval_mapping["description"], EvalTextualMetric)

    # Check list item evaluator
    users_evaluator = evaluator.eval_mapping["users"]
    user_evaluator = users_evaluator.item_evaluator
    assert isinstance(user_evaluator, DictEval)

    # Check user structure
    assert isinstance(user_evaluator.eval_mapping["name"], EvalTextualMetric)
    assert isinstance(user_evaluator.eval_mapping["age"], NumEval)
    assert isinstance(user_evaluator.eval_mapping["metadata"], DictEval)

    # Check metadata structure
    metadata_evaluator = user_evaluator.eval_mapping["metadata"]
    assert isinstance(metadata_evaluator.eval_mapping["created_at"], DateEval)
    assert isinstance(metadata_evaluator.eval_mapping["score"], NumEval)


def test_infer_evaluator_for_empty_list_raises_error() -> None:
    """Test that empty list raises IndexError."""
    data: list[Any] = []

    with pytest.raises(AssertionError, match="List must not be empty to infer evaluator"):
        infer_structured_evaluator_from_predictions(data, text_evaluator="ngram")


def test_infer_evaluator_for_empty_dict() -> None:
    """Test that empty dict returns DictEval with empty mapping."""
    data: dict[str, Any] = {}

    with pytest.raises(AssertionError, match="Dict must not be empty to infer evaluator"):
        infer_structured_evaluator_from_predictions(data, text_evaluator="ngram")


def test_infer_evaluator_for_unsupported_type() -> None:
    """Test that unsupported types raise ValueError."""
    data = set([1, 2, 3])  # set is not supported

    with pytest.raises(
        ValueError, match="Unsupported type encountered during structured evaluator inference"
    ):
        infer_structured_evaluator_from_predictions(data, text_evaluator="ngram")


def test_infer_evaluator_for_none_raises_error() -> None:
    """Test that None type raises ValueError."""
    data = None

    with pytest.raises(
        ValueError, match="Unsupported type encountered during structured evaluator inference"
    ):
        infer_structured_evaluator_from_predictions(data, text_evaluator="ngram")


def test_infer_evaluator_for_boolean_returns_num_eval() -> None:
    """Test that boolean type returns NumEval (since bool is subclass of int)."""
    data = True
    evaluator = infer_structured_evaluator_from_predictions(data, text_evaluator="ngram")

    assert isinstance(evaluator, NumEval)


def test_infer_evaluator_dict_with_mixed_value_types() -> None:
    """Test dict with all supported value types."""
    data = {
        "string_field": "test",
        "int_field": 42,
        "float_field": 3.14,
        "datetime_field": datetime(2021, 1, 1),
        "list_field": ["a", "b"],
        "dict_field": {"nested": "value"},
    }
    evaluator = infer_structured_evaluator_from_predictions(data, text_evaluator="ngram")

    assert isinstance(evaluator, DictEval)
    assert isinstance(evaluator.eval_mapping["string_field"], EvalTextualMetric)
    assert isinstance(evaluator.eval_mapping["int_field"], NumEval)
    assert isinstance(evaluator.eval_mapping["float_field"], NumEval)
    assert isinstance(evaluator.eval_mapping["datetime_field"], DateEval)
    assert isinstance(evaluator.eval_mapping["list_field"], ListEval)
    assert isinstance(evaluator.eval_mapping["dict_field"], DictEval)


def test_infer_evaluator_preserves_list_aggregation_default() -> None:
    """Test that ListEval uses default aggregation setting."""
    data = [1, 2, 3]
    evaluator = infer_structured_evaluator_from_predictions(data, text_evaluator="ngram")

    assert isinstance(evaluator, ListEval)
    assert evaluator.aggregation == DEFAULT_LIST_AGGREGATION
