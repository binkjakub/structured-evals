from sevals.aggregations import AverageAggregation
from sevals.eval_dict import DictEvalOutput


def test_average_aggregation():
    aggregation = AverageAggregation()
    outs = [
        DictEvalOutput(
            results={"a": 1, "b": 2},
            missing={"b": 1},
            extra={},
        ),
        DictEvalOutput(
            results={"a": 2, "b": 4},
            missing={"b": 1},
            extra={"c": 1},
        ),
    ]

    assert aggregation(outs) == {
        "results": {"a": 1.5, "b": 3.0},
        "missing": {"b": 1.0},
        "extra": {"c": 0.5},
    }
