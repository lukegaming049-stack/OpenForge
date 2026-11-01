"""In-process GGUF inference via llama-cpp-python.

Nothing leaves the machine: the weights are mmap'd from ``model_path`` and the
chat template built into the GGUF is used for formatting.

Tool calling
------------
Most GGUF chat templates do not implement the OpenAI ``tools`` contract, so
``tool_mode`` defaults to ``"text"`` here: the agent injects a small text
protocol into the system prompt and parses tool blocks out of the reply.  Set
``tool_mode = "native"`` for templates that do support it (Hermes-2-Pro,
Qwen2.5/Qwen3, Llama-3.1+ with llama-cpp-python >= 0.2.70).
"""

from __future__ import annotations

import threading
from typing import Any

from .base import DeltaCallback, Provider, ProviderError, Turn
from .openai_compat import finalize, merge_stream_delta, new_accumulator

_LOAD_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}

INSTALL_HINT = (
    "GGUF support needs llama-cpp-python.\n"
    "  CPU:          pip install 'openforge[gguf]'\n"
    "  CUDA:         CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python --no-binary :all:\n"
    "  Apple Metal:  CMAKE_ARGS='-DGGML_METAL=on' pip install llama-cpp-python --no-binary :all:\n"
    "  Windows:      pip install llama-cpp-python --extra-index-url "
    "https://abetlen.github.io/llama-cpp-python/whl/cpu"
)


class GGUFProvider(Provider):
    """A locally loaded ``.gguf`` file."""

    def __init__(
        self,
        name: str,
        model: str,
        model_path: str,
        *,
        n_ctx: int = 32768,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        chat_format: str | None = None,
        tool_mode: str = "auto",
        verbose: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name, model)
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.chat_format = chat_format
        self.verbose = verbose
        self.extra = extra or {}
        # "auto" deliberately means text: a template that silently ignores
        # `tools` looks to the user like the agent refusing to act.
        self.tool_mode = "text" if tool_mode == "auto" else tool_mode
        self.supports_native_tools = self.tool_mode == "native"
        self._llm: Any = None

    # ------------------------------------------------------------------ load
    def _cache_key(self) -> tuple[Any, ...]:
        return (
            str(self.model_path),
            self.n_ctx,
            self.n_gpu_layers,
            self.n_threads,
            self.chat_format,
            self.verbose,
        )

    @property
    def llm(self) -> Any:
        if self._llm is not None:
            return self._llm
        key = self._cache_key()
        with _LOAD_LOCK:
            if key in _MODEL_CACHE:
                self._llm = _MODEL_CACHE[key]
                return self._llm
            try:
                from llama_cpp import Llama  # noqa: PLC0415  (optional heavy import)
            except ImportError as exc:  # pragma: no cover - depends on host
                raise ProviderError(INSTALL_HINT) from exc

            kwargs: dict[str, Any] = {
                "model_path": str(self.model_path),
                "n_ctx": self.n_ctx,
                "n_gpu_layers": self.n_gpu_layers,
                "verbose": self.verbose,
            }
            if self.n_threads:
                kwargs["n_threads"] = self.n_threads
            if self.chat_format:
                kwargs["chat_format"] = self.chat_format
            kwargs.update(self.extra)
            try:
                llm = Llama(**kwargs)
            except Exception as exc:  # pragma: no cover - host dependent
                raise ProviderError(f"cannot load {self.model_path}: {exc}") from exc
            _MODEL_CACHE[key] = llm
            self._llm = llm
            return llm

    def close(self) -> None:
        # The cache outlives individual providers on purpose (loading is slow);
        # use unload_all() when you really want the RAM back.
        self._llm = None

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
        llm = self.llm
        wire = self._to_wire_messages(messages)
        kwargs: dict[str, Any] = {
            "messages": wire,
            "temperature": 0.2 if temperature is None else temperature,
            "max_tokens": max_tokens or 2048,
        }
        if tools and self.supports_native_tools:
            kwargs["tools"] = [
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
            kwargs["tool_choice"] = "auto"

        try:
            if on_delta is None:
                data = llm.create_chat_completion(stream=False, **kwargs)
                return self._parse_blocking(data)
            return self._stream(llm, kwargs, on_delta)
        except ProviderError:
            raise
        except Exception as exc:  # pragma: no cover - llama.cpp runtime errors
            raise ProviderError(f"{self.name}: local inference failed: {exc}") from exc

    def _stream(self, llm: Any, kwargs: dict[str, Any], on_delta: DeltaCallback) -> Turn:
        acc = new_accumulator()
        for chunk in llm.create_chat_completion(stream=True, **kwargs):
            if not isinstance(chunk, dict):
                continue
            if chunk.get("usage"):
                acc["usage"] = chunk["usage"]
            for choice in chunk.get("choices") or []:
                before = len(acc["text"])
                merge_stream_delta(acc, choice.get("delta") or {}, choice.get("finish_reason"))
                if len(acc["text"]) > before:
                    on_delta("".join(acc["text"][before:]))
        return finalize(acc)

    def _parse_blocking(self, data: dict[str, Any]) -> Turn:
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(f"{self.name}: local model returned no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        # llama-cpp sometimes hands back a pre-serialised assistant message.
        if not isinstance(message, dict):
            return Turn(text=str(message))
        from .base import ToolCall, parse_tool_arguments

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

    def _to_wire_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content") or ""
            calls = msg.get("tool_calls")
            if role == "tool":
                # Fold tool output into a user turn: text-protocol models expect
                # plain chat roles, and a stray "tool" role breaks most templates.
                wire.append({"role": "user", "content": content})
                continue
            if calls:
                rendered = "\n".join(
                    f"[called {c['name']}({c.get('arguments') or {}})]" for c in calls
                )
                content = (content + "\n" + rendered).strip()
            wire.append({"role": role or "user", "content": content})
        return wire


def unload_all() -> None:
    """Free every cached GGUF model (GPU memory included)."""
    with _LOAD_LOCK:
        _MODEL_CACHE.clear()


def loaded_models() -> list[str]:
    with _LOAD_LOCK:
        return [key[0] for key in _MODEL_CACHE]
