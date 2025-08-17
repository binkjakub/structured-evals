from structured_evals.eval_enum import EnumEval


def test_exact_match() -> None:
    evaluator = EnumEval(["red", "green", "blue"])
    result = evaluator("red", "red")
    assert result.score == 1.0
    assert result.prohibited_value == 0


def test_mismatch_both_allowed() -> None:
    evaluator = EnumEval(["red", "green", "blue"])
    result = evaluator("red", "green")
    assert result.score == 0.0
    assert result.prohibited_value == 0


def test_prohibited_value() -> None:
    evaluator = EnumEval(["red", "green", "blue"])
    result = evaluator("yellow", "red")
    assert result.score == 0.0
    assert result.prohibited_value == 1


def test_none_handling() -> None:
    evaluator = EnumEval(["red", None])
    result = evaluator(None, None)
    assert result.score == 1.0
    assert result.prohibited_value == 0


def test_invalid_type() -> None:
    evaluator = EnumEval(["red", "green"])
    result = evaluator(["red"], "red")  # type: ignore
    assert result.score == 0.0
    assert result.prohibited_value == 0


def test_empty_allowed_values() -> None:
    evaluator = EnumEval([])
    result = evaluator("red", "red")
    assert result.score == 0.0
    assert result.prohibited_value == 1


def test_numeric_types() -> None:
    evaluator = EnumEval([1, 2, 3])
    result = evaluator(1, 1)
    assert result.score == 1.0
    assert result.prohibited_value == 0
