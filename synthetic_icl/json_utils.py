"""Utilities for parsing imperfect JSON emitted by language models."""

from __future__ import annotations

import json
import re
from typing import Any


class RobustJSONParseError(ValueError):
    """Raised when robust_json_parse cannot recover a valid JSON object."""


def _try_json_loads(candidate: str) -> Any:
    return json.loads(candidate.strip())


def _extract_balanced_json(text: str) -> str | None:
    starts = [(idx, char) for idx, char in enumerate(text) if char in "{["]
    pairs = {"{": "}", "[": "]"}
    for start_idx, start_char in starts:
        stack: list[str] = []
        in_string = False
        escape = False
        for idx in range(start_idx, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in pairs:
                stack.append(pairs[char])
            elif stack and char == stack[-1]:
                stack.pop()
                if not stack:
                    return text[start_idx : idx + 1]
    return None


def robust_json_parse(text: str) -> Any:
    """Parse JSON from raw MLLM output with markdown and prose recovery.

    Order:
    1. direct json.loads
    2. fenced ```json ... ``` or generic fenced code block
    3. first balanced JSON object/array in the text
    4. raise a helpful error containing a source snippet
    """
    if not isinstance(text, str):
        raise TypeError(f"robust_json_parse expects str, got {type(text).__name__}.")

    try:
        return _try_json_loads(text)
    except json.JSONDecodeError:
        pass

    fenced_blocks = re.findall(r"```(?:json|JSON)?\s*(.*?)```", text, flags=re.DOTALL)
    for block in fenced_blocks:
        try:
            return _try_json_loads(block)
        except json.JSONDecodeError:
            continue

    balanced = _extract_balanced_json(text)
    if balanced is not None:
        try:
            return _try_json_loads(balanced)
        except json.JSONDecodeError:
            pass

    snippet = text[:1000].replace("\n", "\\n")
    raise RobustJSONParseError(f"Failed to parse JSON from MLLM output. Snippet: {snippet}")
