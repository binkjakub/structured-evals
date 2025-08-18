from datetime import datetime

import pytest

from structured_evals.eval_batch import BatchDictEval
from structured_evals.eval_dict import DictEval
from structured_evals.eval_primitive import DateEval, NumEval


def test_eval_batch() -> None:
    item_evaluator = DictEval(eval_mapping={"num": NumEval(), "date": DateEval()})
    eval_ = BatchDictEval(eval_mapping=item_evaluator.eval_mapping)

    pred = [
        {"num": 1, "date": datetime(2021, 1, 1)},
        {"num": 2, "date": datetime(2021, 1, 1)},
        {"num": 3, "name": "John"},
    ]
    target = [
        {"num": 1, "date": datetime(2021, 1, 1)},
        {"num": 3, "date": datetime(2021, 1, 1)},
        {"num": 3, "date": datetime(2021, 1, 1)},
    ]
    output = eval_(pred, target)
    assert pytest.approx(output.scores, rel=1e-6) == {
        "num": [1.0, 0.0, 1.0],
        "date": [1.0, 1.0, 0.0],
    }
    assert pytest.approx(output.missing_keys, rel=1e-6) == {
        "num": [0.0, 0.0, 0.0],
        "date": [0.0, 0.0, 1.0],
    }
    assert pytest.approx(output.num_times_extra_keys, rel=1e-6) == {"name": 1.0}
