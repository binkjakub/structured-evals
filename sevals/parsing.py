import re
from typing import Any

import yaml

yaml_pattern: re.Pattern = re.compile(r"```(?:ya?ml)?(?P<yaml>[^`]*)", re.MULTILINE | re.DOTALL)


def parse_yaml(text: str) -> dict[str, Any] | None:
    """YAML parser taken from langchain.
    Credit: https://github.com/langchain-ai/langchain.
    """
    match = re.search(yaml_pattern, text.strip())
    if match:
        yaml_str = match.group("yaml")
    else:
        yaml_str = text

    return yaml.safe_load(yaml_str)
