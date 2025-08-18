import pytest

from structured_evals.eval_list import ListEval, ListEvalOutput
from structured_evals.eval_primitive import NumEval


def test_exact_match_same_order() -> None:
    """Test that identical lists in same order get perfect score."""
    evaluator = ListEval(item_evaluator=NumEval())
    result = evaluator([1, 2, 3], [1, 2, 3])

    assert result.score == 1.0
    assert result.num_missing_items == 0
    assert result.num_extra_items == 0


def test_exact_match_different_order() -> None:
    """Test that identical lists in different order get perfect score."""
    evaluator = ListEval(item_evaluator=NumEval())
    result = evaluator([3, 1, 2], [1, 2, 3])

    assert result.score == 1.0
    assert result.num_missing_items == 0
    assert result.num_extra_items == 0


def test_partial_match() -> None:
    """Test partial matching with some correct items."""
    evaluator = ListEval(item_evaluator=NumEval())
    result = evaluator([1, 2, 4], [1, 2, 3])

    assert result.score == pytest.approx(2.0 / 3.0)
    assert result.num_missing_items == 1
    assert result.num_extra_items == 1


def test_no_match() -> None:
    """Test when no items match."""
    evaluator = ListEval(item_evaluator=NumEval())
    result = evaluator([4, 5, 6], [1, 2, 3])

    assert result.score == 0.0
    assert result.num_missing_items == 3
    assert result.num_extra_items == 3


def test_empty_lists() -> None:
    """Test evaluation of empty lists."""
    evaluator = ListEval(item_evaluator=NumEval())
    result = evaluator([], [])

    assert result.score == 1.0
    assert result.num_missing_items == 0
    assert result.num_extra_items == 0


def test_empty_prediction() -> None:
    """Test when prediction is empty but target is not."""
    evaluator = ListEval(item_evaluator=NumEval())
    result = evaluator([], [1, 2, 3])

    assert result.score == 0.0
    assert result.num_missing_items == 3
    assert result.num_extra_items == 0


def test_empty_target() -> None:
    """Test when target is empty but prediction is not."""
    evaluator = ListEval(item_evaluator=NumEval())
    result = evaluator([1, 2, 3], [])

    assert result.score == 0.0
    assert result.num_missing_items == 0
    assert result.num_extra_items == 3


def test_more_predictions_than_targets() -> None:
    """Test when there are more predictions than targets."""
    evaluator = ListEval(item_evaluator=NumEval())
    result = evaluator([1, 2, 3, 4, 5], [1, 2])

    assert result.score == 1.0
    assert result.num_missing_items == 0
    assert result.num_extra_items == 3


def test_more_targets_than_predictions() -> None:
    """Test when there are more targets than predictions."""
    evaluator = ListEval(item_evaluator=NumEval())
    result = evaluator([1, 2], [1, 2, 3, 4, 5])

    assert result.score == pytest.approx(2.0 / 5.0)
    assert result.num_missing_items == 3
    assert result.num_extra_items == 0


def test_duplicate_items_in_prediction() -> None:
    """Test handling of duplicate items in prediction."""
    evaluator = ListEval(item_evaluator=NumEval())
    result = evaluator([1, 1, 2], [1, 2, 3])

    assert result.score == pytest.approx(2.0 / 3.0)
    assert result.num_missing_items == 1
    assert result.num_extra_items == 1


@pytest.mark.skip(reason="Duplication is not supported yet")
def test_duplicate_items_in_target() -> None:
    """Test handling of duplicate items in target."""
    evaluator = ListEval(item_evaluator=NumEval())
    result = evaluator([1, 2, 3], [1, 1, 2])

    assert result.score == 1.0
    assert result.num_missing_items == 0
    assert result.num_extra_items == 1


def test_null_prediction() -> None:
    """Test when prediction is None."""
    evaluator = ListEval(item_evaluator=NumEval())
    result = evaluator(None, [1, 2, 3])  # type: ignore[arg-type]

    assert result.score == 0.0
    assert result.num_missing_items == 3
    assert result.num_extra_items == 0


def test_null_target() -> None:
    """Test when target is None."""
    evaluator = ListEval(item_evaluator=NumEval())
    result = evaluator([1, 2, 3], None)  # type: ignore[arg-type]

    assert result.score == 0.0
    assert result.num_missing_items == 0
    assert result.num_extra_items == 3


def test_both_null() -> None:
    """Test when both prediction and target are None."""
    evaluator = ListEval(item_evaluator=NumEval())
    result = evaluator(None, None)  # type: ignore[arg-type]

    assert result.score == 1.0
    assert result.num_missing_items == 0
    assert result.num_extra_items == 0


def test_string_empty_prediction() -> None:
    """Test when prediction is empty string."""
    evaluator = ListEval(item_evaluator=NumEval())
    result = evaluator("", [1, 2, 3])  # type: ignore[arg-type]

    assert result.score == 0.0
    assert result.num_missing_items == 3
    assert result.num_extra_items == 0


def test_greedy_matching_picks_best() -> None:
    """Test that greedy matching picks the best available match."""

    class MockEvaluator:
        def __call__(self, pred: float, target: float) -> float:
            score = 10.0 - abs(pred - target)
            return type("MockOutput", (), {"score": max(0, score)})()

    evaluator = ListEval(item_evaluator=MockEvaluator())  # type: ignore[arg-type]
    result = evaluator([5, 11], [1, 6])

    assert result.score == pytest.approx((6 + 5) / 2)
    assert result.num_missing_items == 0
    assert result.num_extra_items == 0


def test_aggregation_parameter() -> None:
    """Test that aggregation parameter is stored correctly."""
    evaluator_avg = ListEval(item_evaluator=NumEval(), aggregation="average")
    evaluator_sum = ListEval(item_evaluator=NumEval(), aggregation="sum")

    assert evaluator_avg.aggregation == "average"
    assert evaluator_sum.aggregation == "sum"


def test_zero_score_property() -> None:
    """Test zero_score property returns correct values."""
    evaluator = ListEval(item_evaluator=NumEval())
    zero_score = evaluator.zero_score

    assert isinstance(zero_score, ListEvalOutput)
    assert zero_score.score == 0.0
    assert zero_score.num_missing_items == 0
    assert zero_score.num_extra_items == 0


def test_max_score_property() -> None:
    """Test max_score property returns correct values."""
    evaluator = ListEval(item_evaluator=NumEval())
    max_score = evaluator.max_score

    assert isinstance(max_score, ListEvalOutput)
    assert max_score.score == 1.0
    assert max_score.num_missing_items == 0
    assert max_score.num_extra_items == 0


def test_is_null_method() -> None:
    """Test is_null method recognizes null values correctly."""
    evaluator = ListEval(item_evaluator=NumEval())

    assert evaluator.is_null(None) is True
    assert evaluator.is_null("") is True
    assert evaluator.is_null([]) is True
    assert evaluator.is_null([1]) is False
    assert evaluator.is_null("test") is False
    assert evaluator.is_null(0) is False


def test_check_dtype_method() -> None:
    """Test check_dtype method validates list types correctly."""
    evaluator = ListEval(item_evaluator=NumEval())

    assert evaluator.check_dtype([1, 2], [3, 4]) is True
    assert evaluator.check_dtype([], []) is True
    assert evaluator.check_dtype([1], "not a list") is False  # type: ignore[arg-type]
    assert evaluator.check_dtype("not a list", [1]) is False  # type: ignore[arg-type]
    assert evaluator.check_dtype("not", "lists") is False  # type: ignore[arg-type]


def test_repr_method() -> None:
    """Test string representation of ListEval."""
    evaluator = ListEval(item_evaluator=NumEval(), aggregation="sum")
    repr_str = repr(evaluator)

    assert "ListEval" in repr_str
    assert "aggregation=sum" in repr_str
    assert "NumEval" in repr_str


def test_single_item_lists() -> None:
    """Test evaluation with single-item lists."""
    evaluator = ListEval(item_evaluator=NumEval())

    # Matching single items
    result = evaluator([42], [42])
    assert result.score == 1.0
    assert result.num_missing_items == 0
    assert result.num_extra_items == 0

    # Non-matching single items
    result = evaluator([42], [24])
    assert result.score == 0.0
    assert result.num_missing_items == 1
    assert result.num_extra_items == 1


def test_complex_greedy_scenario() -> None:
    """Test a complex scenario to verify greedy matching behavior."""
    evaluator = ListEval(item_evaluator=NumEval())

    # pred=[1, 2, 2, 3], target=[2, 2, 4]
    # Expected: target[0]=2 matches pred[1]=2, target[1]=2 matches pred[2]=2, target[2]=4 has no match
    # Score: (1.0 + 1.0 + 0.0) / 3 = 2/3
    # Missing: 1 (the 4), Extra: 2 (the 1 and 3)
    result = evaluator([1, 2, 2, 3], [2, 2, 4])

    assert result.score == pytest.approx(2.0 / 3.0)
    assert result.num_missing_items == 1
    assert result.num_extra_items == 2
