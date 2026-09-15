from __future__ import annotations

import json
from typing import Any

from ..log import get_logger
from .base import ErrorAction, StructuredChat, retry_after_seconds

log = get_logger(__name__)


def system_param(system_prompt: str, *, prompt_cache: bool = True) -> Any:
    if prompt_cache:
        return [{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }]
    return system_prompt


def parsed_from_message(message: Any, tool_name: str) -> tuple[dict | None, str]:
    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
            return block.input, json.dumps(block.input, ensure_ascii=False)
    text = "".join(
        getattr(b, "text", "") for b in message.content
        if getattr(b, "type", None) == "text"
    ).strip()
    return None, text


class ClaudeChat(StructuredChat):
    AUTO_MODES = ("tool", "prompt")

    def __init__(
        self,
        *,
        model: str,
        system: str,
        tool: dict | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 2500,
        temperature: float | None = None,
        top_p: float | None = None,
        struct_mode: str = "auto",
        max_retries: int = 5,
        timeout: float = 600.0,
        http_retries: int = 2,
        prompt_cache: bool = True,
    ) -> None:
        super().__init__(struct_mode=struct_mode, max_retries=max_retries)
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError("This stage requires `pip install 'cares[llm]'`") from exc

        kwargs: dict[str, Any] = {"timeout": timeout, "max_retries": http_retries}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = AsyncAnthropic(**kwargs)

        self.model = model
        self.system = system
        self.tool = tool
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.prompt_cache = prompt_cache

    def message_params(self, user: str, mode: str, tool: dict | None = None) -> dict:
        tool = tool or self.tool
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_param(self.system, prompt_cache=self.prompt_cache),
            "messages": [{"role": "user", "content": user}],
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if mode == "tool" and tool is not None:
            params["tools"] = [tool]
            params["tool_choice"] = {"type": "tool", "name": tool["name"]}
        return params

    async def _call(self, prompt: Any, mode: str) -> tuple[dict | None, str]:
        user, tool = prompt if isinstance(prompt, tuple) else (prompt, None)
        tool = tool or self.tool
        resp = await self.client.messages.create(**self.message_params(user, mode, tool))
        if mode == "tool" and tool is not None:
            return parsed_from_message(resp, tool["name"])
        text = "".join(
            getattr(b, "text", "") for b in resp.content
            if getattr(b, "type", None) == "text"
        ).strip()
        return None, text

    def _classify(self, exc: Exception, mode: str, attempt: int) -> tuple[ErrorAction, float, str]:
        import anthropic

        if isinstance(exc, anthropic.BadRequestError):
            return ErrorAction.NEXT_MODE, 0.0, f"bad_request (mode={mode}): {str(exc)[:400]}"
        if isinstance(exc, anthropic.RateLimitError):
            return ErrorAction.RETRY, retry_after_seconds(exc, attempt), f"rate_limited (mode={mode})"
        if isinstance(exc, (anthropic.APIConnectionError, anthropic.InternalServerError,
                            anthropic.APIStatusError)):
            return ErrorAction.RETRY, min(30.0, 1.0 + attempt), f"api_error (mode={mode}): {str(exc)[:200]}"
        return ErrorAction.RETRY, 1.0 + attempt, f"unexpected (mode={mode}): {str(exc)[:200]}"
