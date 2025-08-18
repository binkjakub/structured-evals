from torchmetrics.functional.text import chrf_score


def chrf_eval(pred: str, target: str) -> float:
    return chrf_score([pred], [target], n_char_order=1, n_word_order=0).item()  # type: ignore
