from torchmetrics.functional.text import chrf_score

from sevals.eval_text import EvalTextualMetric


def test_eval_textual_metric() -> None:
    def const_metric(pred: str, target: str) -> float:
        return 0.5

    eval_ = EvalTextualMetric(const_metric, "my_metric")
    assert eval_("pred", "target") == 0.5


def test_eval_textual_chrf() -> None:
    def metric_func(pred: str, target: str) -> float:
        return chrf_score([pred], [target], n_char_order=1, n_word_order=0).item()  # type: ignore[union-attr]

    eval_ = EvalTextualMetric(metric_func, "chrf")
    assert eval_("abc", "def") == 0.0
    assert eval_("abc", "abc") == 1.0
    assert eval_("abcd", "abce") == 0.75
