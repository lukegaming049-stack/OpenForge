"""Anthropic Messages API provider (native tool use, non-streaming).

Kept deliberately small: the request shape differs enough from OpenAI's that
pretending otherwise causes bugs.  Streaming is emulated — the full text is
handed to ``on_delta`` once, so the agent loop stays identical.
"""

from __future__ import annotations

import time
from typing import Any

from .base import DeltaCallback, Provider, ProviderError, ToolCall, Turn

ANTHROPIC_VERSION = "2023-06-01"
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def to_anthropic(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert internal history into (system, anthropic messages)."""
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "system":
            if content:
                system_parts.append(content)
            continue
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id") or "",
                "content": content or "(no output)",
            }
            prev = out[-1] if out else None
            if (
                prev
                and prev["role"] == "user"
                and isinstance(prev["content"], list)
                and all(b.get("type") == "tool_result" for b in prev["content"])
            ):
                prev["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
            continue
        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for tc in msg.get("tool_calls") or []:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc.get("arguments") or {},
                    }
                )
            if blocks:
                out.append({"role": "assistant", "content": blocks})
            continue
        out.append({"role": "user", "content": content})

    # Anthropic rejects a leading assistant turn.
    while out and out[0]["role"] != "user":
        out.pop(0)
    return ("\n\n".join(system_parts) or None), out


class AnthropicProvider(Provider):
    supports_native_tools = True

    def __init__(
        self,
        name: str,
        model: str,
        api_key: str | None,
        base_url: str = "https://api.anthropic.com",
        *,
        timeout: float = 600.0,
        max_retries: int = 3,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name, model)
        if not api_key:
            raise ProviderError(f"{name}: no API key found")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = extra_body or {}

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/v1/messages"

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        on_delta: DeltaCallback | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Turn:
        import httpx

        system, converted = to_anthropic(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": converted,
            "max_tokens": max_tokens or 8192,
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["parameters"],
                }
                for t in tools
            ]
        payload.update(self.extra_body)

        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        timeout = httpx.Timeout(connect=20.0, read=self.timeout, write=60.0, pool=20.0)

        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(self.endpoint, json=payload, headers=headers)
                if resp.status_code in RETRY_STATUS and attempt + 1 < self.max_retries:
                    last = self._error(resp.status_code, resp.text)
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if resp.status_code >= 400:
                    raise self._error(resp.status_code, resp.text)
                turn = self._parse(resp.json())
                if on_delta and turn.text:
                    on_delta(turn.text)
                return turn
            except httpx.TransportError as exc:
                last = ProviderError(f"{self.name}: connection failed: {exc}")
                if attempt + 1 < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise last from exc
        raise last or ProviderError(f"{self.name}: request failed")

    def _error(self, status: int, body: str) -> ProviderError:
        snippet = body.strip()[:600]
        return ProviderError(f"{self.name}: HTTP {status}: {snippet}")

    def _parse(self, data: dict[str, Any]) -> Turn:
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text_parts.append(block.get("text") or "")
            elif block.get("type") == "tool_use":
                calls.append(
                    ToolCall(
                        id=block.get("id") or f"call_{len(calls)}",
                        name=block.get("name") or "",
                        arguments=block.get("input") or {},
                    )
                )
        return Turn(
            text="".join(text_parts),
            tool_calls=[c for c in calls if c.name],
            finish_reason=data.get("stop_reason") or "stop",
            usage=data.get("usage") or {},
        )
