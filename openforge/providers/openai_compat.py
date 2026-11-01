"""OpenAI-compatible chat-completions provider.

Covers OpenAI, OpenRouter, Groq, Together, DeepSeek, Mistral, xAI, Cerebras,
Fireworks, vLLM, Ollama, llama-server and LM Studio — anything that speaks
``POST /chat/completions`` with SSE streaming.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterable, Iterator

from .base import DeltaCallback, Provider, ProviderError, ToolCall, Turn, clean_message, parse_tool_arguments

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


# --------------------------------------------------------------------- parsing
def new_accumulator() -> dict[str, Any]:
    return {"text": [], "tool_calls": {}, "finish_reason": None, "usage": {}}


def merge_stream_delta(
    acc: dict[str, Any], delta: dict[str, Any] | None, finish_reason: str | None = None
) -> None:
    """Fold one streaming ``delta`` object into ``acc`` (mutates in place)."""
    if delta:
        content = delta.get("content")
        if content:
            acc["text"].append(content)
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            slot = acc["tool_calls"].setdefault(
                idx, {"id": None, "name": "", "arguments": ""}
            )
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            name = fn.get("name")
            if name:
                if not slot["name"]:
                    slot["name"] = name
                elif not slot["name"].endswith(name):
                    slot["name"] += name
            if fn.get("arguments"):
                slot["arguments"] += fn["arguments"]
    if finish_reason:
        acc["finish_reason"] = finish_reason


def finalize(acc: dict[str, Any]) -> Turn:
    calls: list[ToolCall] = []
    for idx in sorted(acc["tool_calls"]):
        slot = acc["tool_calls"][idx]
        if not slot["name"]:
            continue
        calls.append(
            ToolCall(
                id=slot["id"] or f"call_{idx}",
                name=slot["name"],
                arguments=parse_tool_arguments(slot["arguments"]),
            )
        )
    return Turn(
        text="".join(acc["text"]),
        tool_calls=calls,
        finish_reason=acc["finish_reason"] or ("tool_calls" if calls else "stop"),
        usage=acc.get("usage") or {},
    )


def iter_sse_json(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield JSON payloads from an SSE line stream, skipping keep-alives."""
    for line in lines:
        if not line:
            continue
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload in ("", "[DONE]"):
            if payload == "[DONE]":
                return
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialise internal history into the wire format."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        m = clean_message(msg)
        calls = m.get("tool_calls")
        if calls:
            m["tool_calls"] = [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {
                        "name": c["name"],
                        "arguments": json.dumps(c.get("arguments") or {}, ensure_ascii=False),
                    },
                }
                for c in calls
            ]
        if m.get("role") == "tool":
            m.pop("name", None)
        out.append(m)
    return out


def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tools
    ]


# -------------------------------------------------------------------- provider
class OpenAICompatProvider(Provider):
    supports_native_tools = True

    def __init__(
        self,
        name: str,
        model: str,
        base_url: str,
        api_key: str | None = None,
        *,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 600.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(name, model)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.extra_headers = extra_headers or {}
        self.extra_body = extra_body or {}
        self.timeout = timeout
        self.max_retries = max_retries

    # ------------------------------------------------------------- internals
    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # OpenRouter likes attribution headers; harmless elsewhere.
        headers.setdefault("HTTP-Referer", "https://github.com/openforge-harness/openforge")
        headers.setdefault("X-Title", "OpenForge")
        return headers

    def _payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        stream: bool,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages),
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if tools:
            payload["tools"] = to_openai_tools(tools)
            payload["tool_choice"] = "auto"
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(self.extra_body)
        return payload

    def _error(self, status: int, body: str) -> ProviderError:
        snippet = body.strip()
        if len(snippet) > 600:
            snippet = snippet[:600] + "…"
        hint = ""
        if status == 401:
            hint = " — check the API key for this provider"
        elif status == 404:
            hint = f" — endpoint {self.endpoint} or model '{self.model}' not found"
        elif status == 429:
            hint = " — rate limited / out of quota"
        return ProviderError(f"{self.name}: HTTP {status}{hint}: {snippet}")

    # ------------------------------------------------------------------ chat
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

        stream = on_delta is not None
        payload = self._payload(messages, tools, stream, temperature, max_tokens)
        timeout = httpx.Timeout(connect=20.0, read=self.timeout, write=60.0, pool=20.0)

        if not stream:
            return self._chat_blocking(httpx, payload, timeout)
        return self._chat_streaming(httpx, payload, timeout, on_delta)

    def _chat_blocking(self, httpx: Any, payload: dict[str, Any], timeout: Any) -> Turn:
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(self.endpoint, json=payload, headers=self._headers())
                if resp.status_code in RETRY_STATUS and attempt + 1 < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    last = self._error(resp.status_code, resp.text)
                    continue
                if resp.status_code >= 400:
                    raise self._error(resp.status_code, resp.text)
                return self._parse_completion(resp.json())
            except (httpx.TransportError,) as exc:
                last = ProviderError(f"{self.name}: connection failed: {exc}")
                if attempt + 1 < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise last from exc
        raise last or ProviderError(f"{self.name}: request failed")

    def _chat_streaming(
        self, httpx: Any, payload: dict[str, Any], timeout: Any, on_delta: DeltaCallback | None
    ) -> Turn:
        acc = new_accumulator()
        try:
            with httpx.Client(timeout=timeout) as client:
                with client.stream(
                    "POST", self.endpoint, json=payload, headers=self._headers()
                ) as resp:
                    if resp.status_code >= 400:
                        raise self._error(resp.status_code, resp.read().decode("utf-8", "replace"))
                    for obj in iter_sse_json(resp.iter_lines()):
                        if obj.get("usage"):
                            acc["usage"] = obj["usage"]
                        for choice in obj.get("choices") or []:
                            delta = choice.get("delta") or {}
                            before = len(acc["text"])
                            merge_stream_delta(acc, delta, choice.get("finish_reason"))
                            if on_delta and len(acc["text"]) > before:
                                on_delta("".join(acc["text"][before:]))
        except httpx.TransportError as exc:
            if acc["text"] or acc["tool_calls"]:
                # Partial answer: better to return what we have than lose it.
                return finalize(acc)
            raise ProviderError(f"{self.name}: connection failed: {exc}") from exc
        return finalize(acc)

    def _parse_completion(self, data: dict[str, Any]) -> Turn:
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(f"{self.name}: response had no choices: {str(data)[:300]}")
        choice = choices[0]
        message = choice.get("message") or {}
        calls = []
        for i, tc in enumerate(message.get("tool_calls") or []):
            fn = tc.get("function") or {}
            if not fn.get("name"):
                continue
            calls.append(
                ToolCall(
                    id=tc.get("id") or f"call_{i}",
                    name=fn["name"],
                    arguments=parse_tool_arguments(fn.get("arguments")),
                )
            )
        return Turn(
            text=message.get("content") or "",
            tool_calls=calls,
            finish_reason=choice.get("finish_reason") or "stop",
            usage=data.get("usage") or {},
        )
