import datetime

from structured_evals.loader import EvaluationBatch, load_jsonl

SAMPLE_JSONL = "data/sample.jsonl"


def test_load_jsonl() -> None:
    data = load_jsonl(SAMPLE_JSONL)
    assert data == [
        {
            "pred": "```yaml\nname: cat\nage: 3\nbirthday: 2020-09-01\n```",
            "target": "```yaml\nname: cat\nage: 3\nbirthday: 2020-09-01\n```",
        },
        {
            "pred": "```yaml\nname: dog\nage: 6\nbirthday: 2018-09-02\n```",
            "target": "```yaml\nname: dog\nage: 5\nbirthday: 2018-09-02\n```",
        },
        {
            "pred": "```yaml\nname: rabbit\nage: 2\nbirthday: 2021-09-05\n```",
            "target": "```yaml\nname: rabbit\nage: 2\nbirthday: 2021-09-03\n```",
        },
    ]


def test_loader_with_parser() -> None:
    eval_batch = EvaluationBatch.from_jsonl(SAMPLE_JSONL)
    pred_data = [
        {"name": "cat", "age": 3, "birthday": datetime.datetime(2020, 9, 1).date()},
        {"name": "dog", "age": 6, "birthday": datetime.datetime(2018, 9, 2).date()},
        {"name": "rabbit", "age": 2, "birthday": datetime.datetime(2021, 9, 5).date()},
    ]

    target_data = [
        {"name": "cat", "age": 3, "birthday": datetime.datetime(2020, 9, 1).date()},
        {"name": "dog", "age": 5, "birthday": datetime.datetime(2018, 9, 2).date()},
        {"name": "rabbit", "age": 2, "birthday": datetime.datetime(2021, 9, 3).date()},
    ]
    assert eval_batch.pred == pred_data
    assert eval_batch.target == target_data
