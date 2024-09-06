from datetime import datetime

import pytest

from sevals.eval_batch import BatchDictEval
from sevals.eval_dict import DictEval
from sevals.eval_primitive import DateEval, NumEval


def test_eval_batch() -> None:
    item_evaluator = DictEval(eval_mapping={"num": NumEval(), "date": DateEval()})
    eval_ = BatchDictEval(item_evaluator=item_evaluator, aggregation="average")

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
    assert pytest.approx(output.agg_results.results, rel=1e-6) == {"num": 2 / 3, "date": 2 / 3}
    assert pytest.approx(output.agg_results.missing, rel=1e-6) == {"date": 1.0}
    assert pytest.approx(output.agg_results.extra) == {"name": 1.0}
