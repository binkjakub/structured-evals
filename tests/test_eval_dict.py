from datetime import datetime

import pytest

from sevals import DictEvalOutput
from sevals.eval_dict import DictEval
from sevals.eval_primitive import DateEval, NumEval


def test_eval_dict_with_primitives_all_correct() -> None:
    eval_ = DictEval(eval_mapping={"num": NumEval(), "date": DateEval()})

    pred = {"num": 1, "date": datetime(2021, 1, 1)}
    target = {"num": 1, "date": datetime(2021, 1, 1)}
    output = eval_(pred, target)
    assert output.results == {"num": 1.0, "date": 1.0}
    assert output.missing == {}
    assert output.extra == {}


def test_eval_dict_with_primitives_mismatch_single_field() -> None:
    eval_ = DictEval(eval_mapping={"num": NumEval(), "date": DateEval()})

    pred = {"num": 1, "date": datetime(2021, 1, 1)}
    target = {"num": 2, "date": datetime(2021, 1, 1)}
    output = eval_(pred, target)
    assert output.results == {"num": 0.0, "date": 1.0}
    assert output.missing == {}
    assert output.extra == {}


def test_eval_dict_with_primitives_extra_field() -> None:
    eval_ = DictEval(
        eval_mapping={
            "num": NumEval(),
        }
    )

    pred = {"num": 1, "date": datetime(2021, 1, 1)}
    target = {
        "num": 1,
    }
    output = eval_(pred, target)
    assert output.results == {"num": 1.0}
    assert output.missing == {}
    assert output.extra == {"date": 1.0}


def test_eval_dict_raises_on_extra_field_in_target() -> None:
    eval_ = DictEval(
        eval_mapping={
            "num": NumEval(),
        }
    )

    pred = {
        "num": 1,
    }
    target = {"num": 1, "date": datetime(2021, 1, 1)}
    with pytest.raises(ValueError):
        eval_(pred, target)


def test_eval_dict_raises_on_default_strategy_and_invalid_field_dtype() -> None:
    eval_ = DictEval(
        eval_mapping={
            "num": NumEval(),
        }
    )

    pred = {
        "num": 1,
    }
    target = {"num": "1"}
    with pytest.raises(TypeError):
        eval_(pred, target)


def test_eval_dict_ignore_error_strategy_and_invalid_field_dtype() -> None:
    eval_ = DictEval(
        eval_mapping={
            "num": NumEval(),
        },
        error_strategy="ignore",
    )

    pred = {
        "num": 1,
    }
    target = {"num": "1"}
    output = eval_(pred, target)
    assert output.results == {"num": 0.0}
    assert output.missing == {}
    assert output.extra == {}


def test_dict_eval_output_raises_on_conflicting_fields() -> None:
    try:
        DictEvalOutput(
            results={"a": 1, "b": 0},
            missing={"b": 1},
            extra={"b": 1},
        )
    except ValueError:
        pytest.fail("DictEvalOutput should not raise on valid input")

    with pytest.raises(ValueError):
        DictEvalOutput(
            results={"a": 1, "b": 1},
            missing={"b": 1},
            extra={"b": 1},
        )
