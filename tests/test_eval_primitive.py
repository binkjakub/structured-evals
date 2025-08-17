from datetime import datetime

import pytest

from structured_evals.eval_primitive import DateEval, NumEval


def test_eval_int() -> None:
    evaluator = NumEval()
    assert evaluator(1, 1).score == 1.0
    assert evaluator(1, 2).score == 0.0


def test_eval_float() -> None:
    evaluator = NumEval()
    assert evaluator(1.0, 1.0).score == 1.0
    assert evaluator(1.0, 2.0).score == 0.0


def test_num_eval_with_bad_types() -> None:
    evaluator = NumEval()
    with pytest.raises(TypeError):
        evaluator(1, "target")  # type: ignore


def test_date_eval_equal_dates() -> None:
    evaluator = DateEval()
    assert evaluator(datetime(2021, 1, 1), datetime(2021, 1, 1)).score == 1.0
    assert evaluator(datetime(2021, 1, 1), datetime(2021, 1, 2)).score == 0.0


def test_date_eval_ignores_time_part() -> None:
    evaluator = DateEval()
    assert evaluator(datetime(2021, 1, 1, 12, 34, 56), datetime(2021, 1, 1)).score == 1.0
    assert (
        evaluator(datetime(2021, 1, 1, 12, 34, 56), datetime(2021, 1, 1, 10, 45, 11)).score == 1.0
    )
    assert evaluator(datetime(2021, 1, 1, 12, 34, 56), datetime(2021, 1, 2)).score == 0.0


def test_date_equal_invalid_dtype() -> None:
    evaluator = DateEval()
    with pytest.raises(TypeError):
        evaluator(datetime(2021, 1, 1), "2021-01-01")  # type: ignore
