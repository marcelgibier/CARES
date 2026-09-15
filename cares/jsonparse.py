from __future__ import annotations

import json
import re
from typing import Any

_LANG_TAG = re.compile(r"^[A-Za-z0-9_+-]*[ \t]*")


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    body = cleaned[3:]
    if "\n" in body:
        body = body.split("\n", 1)[1]
    else:
        body = _LANG_TAG.sub("", body, count=1)
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body


def _extract_balanced(text: str, opener: str, closer: str) -> str | None:
    start = text.find(opener)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json_object(text: str) -> str | None:
    if not text:
        return None
    return _extract_balanced(_strip_code_fence(text), "{", "}")


def extract_json_array(text: str) -> str | None:
    if not text:
        return None
    return _extract_balanced(_strip_code_fence(text), "[", "]")


def _balanced_candidates(text: str) -> list[str]:
    cleaned = _strip_code_fence(text)
    found: list[tuple[int, str]] = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        if start == -1:
            continue
        snippet = _extract_balanced(cleaned, opener, closer)
        if snippet:
            found.append((start, snippet))
    return [snippet for _, snippet in sorted(found, key=lambda pair: pair[0])]


def extract_json_value(text: str) -> str | None:
    if not text:
        return None
    candidates = _balanced_candidates(text)
    return candidates[0] if candidates else None


def parse_json_loose(text: str) -> Any | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for snippet in _balanced_candidates(text):
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue
    return None


def find_list(parsed: Any, aliases: tuple[str, ...]) -> list | None:
    if isinstance(parsed, list):
        return parsed
    if not isinstance(parsed, dict):
        return None
    for key in aliases:
        value = parsed.get(key)
        if isinstance(value, list):
            return value
    list_keys = [k for k, v in parsed.items() if isinstance(v, list)]
    if len(list_keys) == 1:
        return parsed[list_keys[0]]
    return None
