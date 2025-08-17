import datetime

from structured_evals.loader import EvaluationBatch, load_jsonl

SAMPLE_JSONL = "data/sample.jsonl"
SAMPLE_JSON = "data/sample.json"


def test_load_jsonl() -> None:
    data = load_jsonl(SAMPLE_JSONL)
    assert data == [
        {
            "pred": '```json\n{\n  "name": "cat",\n  "age": 3,\n  "birthday": "2020-09-01"\n}\n```',
            "target": '```json\n{\n  "name": "cat",\n  "age": 3,\n  "birthday": "2020-09-01"\n}\n```',
        },
        {
            "pred": '```json\n{\n  "name": "dog",\n  "age": 6,\n  "birthday": "2018-09-02"\n}\n```',
            "target": '```json\n{\n  "name": "dog",\n  "age": 5,\n  "birthday": "2018-09-02"\n}\n```',
        },
        {
            "pred": '```json\n{\n  "name": "rabbit",\n  "age": 2,\n  "birthday": "2021-09-05"\n}\n```',
            "target": '```json\n{\n  "name": "rabbit",\n  "age": 2,\n  "birthday": "2021-09-03"\n}\n```',
        },
    ]


def test_loader_with_parser() -> None:
    eval_batch = EvaluationBatch.from_json(SAMPLE_JSON, record_format="json")
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
