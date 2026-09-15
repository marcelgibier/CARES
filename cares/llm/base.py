from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..jsonparse import extract_json_value
from ..log import get_logger

log = get_logger(__name__)

Postprocess = Callable[[Any], "tuple[Any | None, str]"]


def retry_after_seconds(exc: Exception, attempt: int) -> float:
    try:
        header = exc.response.headers.get("retry-after")  # type: ignore
        if header is not None:
            return float(header)
    except Exception:  # noqa: BLE001
        pass
    return min(60.0, 2.0**attempt)


class ErrorAction(Enum):
    RETRY = "retry"
    NEXT_MODE = "next_mode"
    ABORT = "abort"


@dataclass
class GenerationResult:
    value: Any | None
    raw: str
    status: str

    @property
    def ok(self) -> bool:
        return self.value is not None


def parse_response(raw: str, mode: str) -> tuple[Any | None, str]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    else:
        if parsed is not None:
            return parsed, ""
        return None, f"json_null (mode={mode})"
    snippet = extract_json_value(raw)
    if not snippet:
        return None, f"no_json_value_found (mode={mode})"
    try:
        return json.loads(snippet), ""
    except json.JSONDecodeError as exc:
        return None, f"json_parse_failed (mode={mode}): {exc}"


class StructuredChat(ABC):
    AUTO_MODES: tuple[str, ...] = ()

    def __init__(self, *, struct_mode: str = "auto", max_retries: int = 5) -> None:
        self.struct_mode = struct_mode
        self.max_retries = max_retries

    @abstractmethod
    async def _call(self, prompt: Any, mode: str) -> tuple[Any | None, str]:
        pass

    @abstractmethod
    def _classify(self, exc: Exception, mode: str, attempt: int) -> tuple[ErrorAction, float, str]:
        pass

    def _modes(self) -> list[str]:
        if self.struct_mode == "auto":
            return list(self.AUTO_MODES)
        return [self.struct_mode]

    async def generate(
        self,
        prompt: Any,
        postprocess: Postprocess,
        *,
        label: str = "",
        max_retries: int | None = None,
        struct_mode: str | None = None,
    ) -> GenerationResult:
        retries = self.max_retries if max_retries is None else max_retries
        modes = [struct_mode] if struct_mode else self._modes()
        tag = f"'{label}' " if label else ""
        last_raw = ""
        last_reason = "no_attempt"

        for mode in modes:
            for attempt in range(retries):
                try:
                    parsed, raw = await self._call(prompt, mode)
                except Exception as exc:  # noqa: BLE001
                    action, wait, reason = self._classify(exc, mode, attempt)
                    last_reason = reason
                    if action is ErrorAction.ABORT:
                        log.error("  %sgiving up: %s", tag, reason)
                        return GenerationResult(None, last_raw, f"aborted: {reason}")
                    if action is ErrorAction.NEXT_MODE:
                        log.warning("  %smode '%s' rejected -> falling back (%s)", tag, mode, reason)
                        break
                    log.warning("  %sAPI error (mode=%s, attempt %d): %s",
                                tag, mode, attempt + 1, reason)
                    if wait > 0:
                        await asyncio.sleep(wait)
                    continue

                last_raw = raw

                if parsed is None:
                    if not raw:
                        last_reason = f"empty_response (mode={mode})"
                        log.warning("  %sempty response (mode=%s, attempt %d)",
                                    tag, mode, attempt + 1)
                        continue
                    parsed, reason = parse_response(raw, mode)
                    if parsed is None:
                        last_reason = reason
                        continue

                value, reason = postprocess(parsed)
                if value is not None:
                    return GenerationResult(value, raw, f"ok (mode={mode})")

                last_reason = f"{reason} (mode={mode})"
                log.info("  attempt %d/%d %s(mode=%s): %s",
                         attempt + 1, retries, tag, mode, reason)

        return GenerationResult(None, last_raw, f"validation_failed: {last_reason}")
