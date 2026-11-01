"""Provider abstraction.

Every provider speaks one internal message format (OpenAI-ish dicts) and returns
a :class:`Turn`.  The agent never knows which vendor answered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


class ProviderError(RuntimeError):
    """Any provider-side failure, with a message safe to show a human."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_message_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class Turn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, Any] = field(default_factory=dict)


DeltaCallback = Callable[[str], None]


class Provider:
    """Base class.  Subclasses implement :meth:`chat`."""

    #: True when the endpoint understands OpenAI/Anthropic style tool schemas.
    supports_native_tools: bool = True

    def __init__(self, name: str, model: str) -> None:
        self.name = name
        self.model = model

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        on_delta: DeltaCallback | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Turn:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        """Release sockets / GPU memory."""

    def __enter__(self) -> "Provider":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}:{self.model}>"


def parse_tool_arguments(raw: Any) -> dict[str, Any]:
    """Models emit arguments as a JSON string, a dict, or garbage. Normalise."""
    if isinstance(raw, dict):
        return raw
    if raw in (None, ""):
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"__raw__": raw}
        if isinstance(parsed, dict):
            return parsed
        return {"__raw__": raw}
    return {"__raw__": str(raw)}


def clean_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Drop empty optional keys so strict endpoints don't reject the payload."""
    out = dict(msg)
    if out.get("tool_calls") == []:
        out.pop("tool_calls")
    if "content" not in out:
        out["content"] = ""
    return out
