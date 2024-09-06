from datetime import datetime

from sevals.eval_primitive import DateEval, NumEval


def test_eval_int() -> None:
    eval_ = NumEval()
    assert eval_.evaluate(1, 1) == 1.0
    assert eval_.evaluate(1, 2) == 0.0


def test_eval_float() -> None:
    eval_ = NumEval()
    assert eval_.evaluate(1.0, 1.0) == 1.0
    assert eval_.evaluate(1.0, 2.0) == 0.0


def test_date_equal() -> None:
    eval_ = DateEval()
    assert eval_.evaluate(datetime(2021, 1, 1), datetime(2021, 1, 1)) == 1.0
    assert eval_.evaluate(datetime(2021, 1, 1), datetime(2021, 1, 2)) == 0.0


def test_date_eval_ignores_time_part() -> None:
    eval_ = DateEval()
    assert eval_.evaluate(datetime(2021, 1, 1, 12, 34, 56), datetime(2021, 1, 1)) == 1.0
    assert eval_.evaluate(datetime(2021, 1, 1, 12, 34, 56), datetime(2021, 1, 1, 10, 45, 11)) == 1.0
    assert eval_.evaluate(datetime(2021, 1, 1, 12, 34, 56), datetime(2021, 1, 2)) == 0.0
