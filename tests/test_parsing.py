import datetime

from sevals.parsing import parse_yaml


def test_parses_valid_yaml() -> None:
    text = "```yaml\nkey: value\n```"
    result = parse_yaml(text)
    assert result == {"key": "value"}


def test_returns_none_for_empty_string() -> None:
    text = ""
    result = parse_yaml(text)
    assert result is None


def test_valid_yaml_without_fence() -> None:
    text = "key: value"
    result = parse_yaml(text)
    assert result == {"key": "value"}


def test_handles_no_yaml_content() -> None:
    text = "```yaml\n\n```"
    result = parse_yaml(text)
    assert result is None


def test_handles_dates() -> None:
    text = "```yaml\nkey: 2022-01-01\n```"
    result = parse_yaml(text)
    assert result == {"key": datetime.date(2022, 1, 1)}
