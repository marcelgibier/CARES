from __future__ import annotations

import random
from typing import Any

from ..log import get_logger
from .base import ErrorAction, StructuredChat, retry_after_seconds

log = get_logger(__name__)

_MODE_REJECTED = ("unsupported", "unknown", "invalid", "bad request", "400", "not supported")

_STRUCT_PARAMS = ("response_format", "guided_json", "extra_body")


def _rejected_param(message: str) -> str | None:
    marker = "'param':"
    i = message.find(marker)
    if i < 0:
        return None
    rest = message[i + len(marker):].lstrip()
    if rest[:1] not in ("'", '"'):
        return None
    end = rest.find(rest[0], 1)
    return rest[1:end] if end > 1 else None


class OpenAIChat(StructuredChat):
    AUTO_MODES = ("response_format", "guided_json", "json_object")

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        schema: dict | None = None,
        schema_name: str = "response",
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        struct_mode: str = "auto",
        max_retries: int = 5,
        timeout: float = 600.0,
        http_retries: int = 3,
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
    ) -> None:
        super().__init__(struct_mode=struct_mode, max_retries=max_retries)
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("This stage requires `pip install 'cares[llm]'`") from exc

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=http_retries,
        )
        self.model = model
        self.schema = schema
        self.schema_name = schema_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.service_tier = service_tier

    def request_params(self, messages: list[dict], mode: str, seed: int | None = None) -> dict:
        params: dict[str, Any] = {"model": self.model, "messages": messages}
        reasoning = self.reasoning_effort is not None
        if self.temperature is not None and not reasoning:
            params["temperature"] = self.temperature
        if self.top_p is not None and not reasoning:
            params["top_p"] = self.top_p
        if self.max_tokens is not None:
            params["max_completion_tokens" if reasoning else "max_tokens"] = self.max_tokens
        if seed is not None:
            params["seed"] = seed

        if self.schema is not None:
            if mode == "response_format":
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.schema_name,
                        "schema": self.schema,
                        "strict": False,
                    },
                }
            elif mode == "guided_json":
                params["extra_body"] = {"guided_json": self.schema}
        if mode == "json_object":
            params["response_format"] = {"type": "json_object"}

        extra = {k: v for k, v in (("reasoning_effort", self.reasoning_effort),
                                   ("service_tier", self.service_tier)) if v is not None}
        if extra:
            params["extra_body"] = {**params.get("extra_body", {}), **extra}
        return params

    async def _call(self, prompt: Any, mode: str) -> tuple[None, str]:
        seed = random.randint(0, 2**31 - 1)
        params = self.request_params(prompt, mode, seed=seed)
        resp = await self.client.chat.completions.create(**params)
        return None, (resp.choices[0].message.content or "").strip()

    def _classify(self, exc: Exception, mode: str, attempt: int) -> tuple[ErrorAction, float, str]:
        import openai

        if isinstance(exc, openai.RateLimitError):
            return ErrorAction.RETRY, retry_after_seconds(exc, attempt), f"rate_limited (mode={mode})"
        msg = str(exc)
        param = _rejected_param(msg)
        if param is not None and param not in _STRUCT_PARAMS:
            return ErrorAction.ABORT, 0.0, f"parameter '{param}' refused by the API: {msg[:300]}"
        if any(s in msg.lower() for s in _MODE_REJECTED):
            return ErrorAction.NEXT_MODE, 0.0, f"mode '{mode}' rejected: {msg[:200]}"
        return ErrorAction.RETRY, 1.0 + attempt, f"api_error ({mode}): {msg[:200]}"
