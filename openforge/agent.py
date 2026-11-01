"""The agent loop.

    user prompt -> model -> (tool calls -> results)* -> final answer

Provider-agnostic: native tool calling and the text protocol both end up in the
same list of :class:`ToolCall` objects.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Config
from .prompts import build_system_prompt
from .providers.base import Provider, ProviderError, ToolCall, Turn
from .tools import ToolContext, execute, schemas, tool_doc

# ```tool ... ``` blocks (also tolerates ```json blocks that carry a tool call)
TOOL_BLOCK_RE = re.compile(r"```(?:tool|json)[ \t]*\r?\n(.*?)```", re.DOTALL)
MENTION_RE = re.compile(r"@([A-Za-z0-9_./\-]+)")
MAX_REPEATS = 3


@dataclass
class AgentEvent:
    """Something worth showing the user. Emitted synchronously."""

    kind: str  # delta | assistant | tool_start | tool_result | notice | error | done
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)


EventHandler = Callable[[AgentEvent], None]


# ---------------------------------------------------------------- text protocol
def parse_tool_blocks(text: str) -> tuple[str, list[ToolCall]]:
    """Split a text-protocol reply into (prose, tool calls)."""
    calls: list[ToolCall] = []

    def collect(payload: str) -> None:
        payload = payload.strip()
        if not payload:
            return
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            # tolerate trailing commas / prose around the object
            start, end = payload.find("{"), payload.rfind("}")
            if start == -1 or end <= start:
                return
            try:
                obj = json.loads(payload[start : end + 1])
            except json.JSONDecodeError:
                return
        items = obj if isinstance(obj, list) else [obj]
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("tool")
            if not isinstance(name, str) or not name:
                continue
            args = item.get("arguments", item.get("args", item.get("parameters", {})))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"__raw__": args}
            if not isinstance(args, dict):
                args = {}
            calls.append(ToolCall(id=f"call_{len(calls)}", name=name, arguments=args))

    for match in TOOL_BLOCK_RE.finditer(text):
        collect(match.group(1))

    cleaned = TOOL_BLOCK_RE.sub("", text).strip()
    return cleaned, calls


# ---------------------------------------------------------------------- agent
class Agent:
    def __init__(
        self,
        provider: Provider,
        model: str,
        ctx: ToolContext,
        *,
        config: Config | None = None,
        tool_mode: str = "auto",
        max_steps: int | None = None,
        temperature: float | None = None,
        max_tool_output: int | None = None,
        on_event: EventHandler | None = None,
        system_prompt: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.ctx = ctx
        cfg = config or Config()
        self.config = cfg
        self.max_steps = max_steps or cfg.max_steps
        self.temperature = cfg.temperature if temperature is None else temperature
        self.max_tool_output = max_tool_output or cfg.max_tool_output_chars
        self.on_event = on_event
        self.messages: list[dict[str, Any]] = list(history or [])
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.steps_taken = 0

        resolved_mode = tool_mode
        if resolved_mode == "auto":
            resolved_mode = "native" if provider.supports_native_tools else "text"
        if resolved_mode == "native" and not provider.supports_native_tools:
            resolved_mode = "text"
        self.tool_mode = resolved_mode

        self.system_prompt = system_prompt or build_system_prompt(
            str(ctx.root),
            tool_mode=self.tool_mode,
            tool_list=tool_doc(),
            extra=self._project_notes(),
        )

    # ------------------------------------------------------------------ helpers
    def _project_notes(self) -> str:
        """Pick up AGENTS.md / OPENFORGE.md so the agent follows repo conventions."""
        for name in ("OPENFORGE.md", "AGENTS.md", ".openforge.md"):
            candidate = self.ctx.root / name
            if candidate.is_file():
                try:
                    text = candidate.read_text(encoding="utf-8")
                except OSError:
                    continue
                if len(text) > 8000:
                    text = text[:8000] + "\n... (truncated)"
                return f"(from {name})\n\n{text}"
        return ""

    def _emit(self, kind: str, text: str = "", **data: Any) -> None:
        if self.on_event:
            self.on_event(AgentEvent(kind=kind, text=text, data=data))

    def _wire_messages(self) -> list[dict[str, Any]]:
        return [{"role": "system", "content": self.system_prompt}, *self.messages]

    def _expand_mentions(self, prompt: str) -> str:
        """`@path/to/file` in a prompt attaches the file as context."""
        attached: list[str] = []

        def replace(match: re.Match[str]) -> str:
            rel = match.group(1).rstrip(".,;)")
            try:
                target = self.ctx.resolve(rel, must_exist=True)
            except Exception:
                return match.group(0)
            if not target.is_file():
                return match.group(0)
            try:
                body = target.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return match.group(0)
            if len(body) > 20_000:
                body = body[:20_000] + "\n... (truncated)"
            attached.append(f"### {self.ctx.relative(target)}\n```\n{body}\n```")
            return self.ctx.relative(target)

        cleaned = MENTION_RE.sub(replace, prompt)
        if attached:
            return cleaned + "\n\nFiles I attached:\n\n" + "\n\n".join(attached)
        return cleaned

    def _maybe_compact(self) -> None:
        """Cheap context guard: elide old tool output instead of failing the call."""
        budget = 400_000
        total = sum(len(str(m.get("content") or "")) for m in self.messages)
        if total <= budget:
            return
        elided = 0
        for msg in self.messages[:-6]:
            content = str(msg.get("content") or "")
            if msg.get("role") in ("tool", "user") and len(content) > 400:
                msg["content"] = content[:200] + "\n... [older output elided to save context]"
                elided += 1
        if elided:
            self._emit("notice", f"compacted {elided} old tool result(s) to stay inside the context window")

    # --------------------------------------------------------------------- run
    def run(self, prompt: str) -> str:
        """Run to completion and return the final assistant text."""
        self.messages.append({"role": "user", "content": self._expand_mentions(prompt)})
        repeats: dict[tuple[str, str], int] = {}

        for step in range(1, self.max_steps + 1):
            self.steps_taken = step
            self._maybe_compact()
            try:
                turn = self._chat()
            except ProviderError as exc:
                self._emit("error", str(exc))
                raise

            self._record_usage(turn)
            prose, calls = self._interpret(turn)

            if not calls:
                self.messages.append({"role": "assistant", "content": prose})
                self._emit("assistant", prose)
                self._emit("done", prose, steps=step)
                return prose

            self.messages.append(
                {
                    "role": "assistant",
                    "content": prose,
                    "tool_calls": [c.to_message_dict() for c in calls],
                }
            )

            results: list[tuple[str, str, bool]] = []
            for call in calls:
                key = (call.name, json.dumps(call.arguments, sort_keys=True, default=str))
                repeats[key] = repeats.get(key, 0) + 1
                if repeats[key] > MAX_REPEATS:
                    message = (
                        f"Stopped: the model called {call.name} with identical arguments "
                        f"{repeats[key]} times. Tell it what to do differently."
                    )
                    self._emit("error", message)
                    return message

                self._emit("tool_start", data={"name": call.name, "arguments": call.arguments})
                output, is_error = execute(
                    self.ctx, call.name, call.arguments, max_output=self.max_tool_output
                )
                self._emit(
                    "tool_result",
                    text=output,
                    name=call.name,
                    is_error=is_error,
                    arguments=call.arguments,
                )
                results.append((call.name, output, is_error))

            self._append_results(calls, results)

        message = (
            f"Stopped after {self.max_steps} steps without finishing. "
            "Raise max_steps in config.toml, or split the task into smaller asks."
        )
        self._emit("error", message)
        return message

    def _chat(self) -> Turn:
        streaming = self.on_event is not None
        tool_schemas = schemas() if self.tool_mode == "native" else None

        def on_delta(chunk: str) -> None:
            self._emit("delta", chunk)

        return self.provider.chat(
            self._wire_messages(),
            tool_schemas,
            on_delta=on_delta if streaming else None,
            temperature=self.temperature,
        )

    def _interpret(self, turn: Turn) -> tuple[str, list[ToolCall]]:
        if self.tool_mode == "text":
            return parse_tool_blocks(turn.text)
        return turn.text, turn.tool_calls

    def _append_results(
        self, calls: list[ToolCall], results: list[tuple[str, str, bool]]
    ) -> None:
        if self.tool_mode == "native":
            for call, (_name, output, _is_error) in zip(calls, results):
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": output,
                    }
                )
            return
        blocks = [
            f"### {name} {'(error)' if is_error else ''}\n```\n{output}\n```"
            for name, output, is_error in results
        ]
        self.messages.append(
            {"role": "user", "content": "Tool results:\n\n" + "\n\n".join(blocks)}
        )

    def _record_usage(self, turn: Turn) -> None:
        usage = turn.usage or {}
        self.usage["prompt_tokens"] += int(
            usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        )
        self.usage["completion_tokens"] += int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )

    # ------------------------------------------------------------------ extras
    def reset(self) -> None:
        self.messages.clear()
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.steps_taken = 0

    def export(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider.name,
            "tool_mode": self.tool_mode,
            "root": str(self.ctx.root),
            "messages": self.messages,
        }

    def load(self, data: dict[str, Any]) -> None:
        self.messages = list(data.get("messages") or [])


def make_context(
    root: pathlib.Path | str,
    *,
    approve: Callable[..., bool] | None = None,
    allow_outside_root: bool = False,
    dry_run: bool = False,
    on_change: Callable[[pathlib.Path], None] | None = None,
) -> ToolContext:
    return ToolContext(
        root=pathlib.Path(root).expanduser().resolve(),
        approve=approve or (lambda request: False),
        allow_outside_root=allow_outside_root,
        dry_run=dry_run,
        on_change=on_change,
    )
