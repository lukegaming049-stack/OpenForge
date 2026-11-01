"""Configuration and BYOK credential resolution for OpenForge.

Everything is bring-your-own-key: there are no accounts, no hosted proxy, no
telemetry.  A provider is usable only if it can find a credential, either in
``~/.config/openforge/keys.toml`` or in the environment variable named by
``api_key_env``.

Resolution order for a provider's key:
    1. ``keys.toml``            [keys] <provider> = "sk-..."
    2. ``config.toml``          [providers.<name>] api_key = "..."   (discouraged)
    3. environment              $<api_key_env>  (or the preset's default var)

Local providers (GGUF, Ollama, llama-server, LM Studio) need no key at all.
"""

from __future__ import annotations

import os
import pathlib
import tomllib
from dataclasses import dataclass, field
from typing import Any

APP_NAME = "openforge"

#: Known provider endpoints.  A ``[providers.<name>]`` table may inherit one of
#: these with ``preset = "<name>"`` or simply by using the preset's own name.
PRESETS: dict[str, dict[str, Any]] = {
    "openai": {
        "type": "openai_compat",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4.1-mini",
    },
    "anthropic": {
        "type": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4-20250514",
    },
    "openrouter": {
        "type": "openai_compat",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "anthropic/claude-sonnet-4",
    },
    "groq": {
        "type": "openai_compat",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "cerebras": {
        "type": "openai_compat",
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "default_model": "llama-3.3-70b",
    },
    "together": {
        "type": "openai_compat",
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "default_model": "Qwen/Qwen2.5-Coder-32B-Instruct",
    },
    "deepseek": {
        "type": "openai_compat",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
    },
    "mistral": {
        "type": "openai_compat",
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "default_model": "mistral-large-latest",
    },
    "xai": {
        "type": "openai_compat",
        "base_url": "https://api.x.ai/v1",
        "api_key_env": "XAI_API_KEY",
        "default_model": "grok-3-mini",
    },
    "fireworks": {
        "type": "openai_compat",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "default_model": "accounts/fireworks/models/qwen3-coder-480b-a35b-instruct",
    },
    "ollama": {
        "type": "openai_compat",
        "base_url": "http://localhost:11434/v1",
        "api_key_env": None,
        "default_model": "qwen2.5-coder:7b",
    },
    "llama-server": {
        "type": "openai_compat",
        "base_url": "http://localhost:8080/v1",
        "api_key_env": None,
        "default_model": "local",
    },
    "lmstudio": {
        "type": "openai_compat",
        "base_url": "http://localhost:1234/v1",
        "api_key_env": None,
        "default_model": "local-model",
    },
}

#: Presets whose credential is simply "is the env var set".
_KEYED_PRESETS = [n for n, p in PRESETS.items() if p.get("api_key_env")]


class ConfigError(RuntimeError):
    """Raised when configuration is missing or unusable."""


def config_dir() -> pathlib.Path:
    override = os.environ.get("OPENFORGE_CONFIG_DIR")
    if override:
        return pathlib.Path(override).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME")
    root = pathlib.Path(base).expanduser() if base else pathlib.Path.home() / ".config"
    return root / APP_NAME


def config_path() -> pathlib.Path:
    return config_dir() / "config.toml"


def keys_path() -> pathlib.Path:
    return config_dir() / "keys.toml"


def redact(secret: str | None) -> str:
    if not secret:
        return "<none>"
    if len(secret) <= 10:
        return secret[:2] + "***"
    return f"{secret[:6]}…{secret[-4:]}"


@dataclass
class ProviderConfig:
    """One configured endpoint."""

    name: str
    type: str = "openai_compat"  # openai_compat | anthropic | gguf
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    model_path: str | None = None
    default_model: str | None = None
    # GGUF-specific
    n_ctx: int = 32768
    n_gpu_layers: int = -1
    n_threads: int | None = None
    chat_format: str | None = None
    # "auto" -> native tool calling where the provider supports it, else text
    tool_mode: str = "auto"
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ keys
    def env_key(self) -> str | None:
        if self.api_key_env is False:
            return None
        if self.api_key_env:
            return os.environ.get(self.api_key_env) or None
        return None

    def resolve_key(self, keys_file: dict[str, str] | None = None) -> str | None:
        if self.type == "gguf":
            return None
        if keys_file and keys_file.get(self.name):
            return keys_file[self.name]
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env) or None
        return None

    def key_source(self, keys_file: dict[str, str] | None = None) -> str:
        if self.type == "gguf":
            return "local model"
        if keys_file and keys_file.get(self.name):
            return "keys.toml"
        if self.api_key:
            return "config.toml"
        if self.api_key_env and os.environ.get(self.api_key_env):
            return f"${self.api_key_env}"
        return "missing"

    def resolved_model_path(self) -> pathlib.Path:
        if not self.model_path:
            raise ConfigError(f"provider '{self.name}' is type=gguf but has no model_path")
        return pathlib.Path(self.model_path).expanduser()

    def describe(self, keys_file: dict[str, str] | None = None) -> str:
        if self.type == "gguf":
            return f"gguf {self.model_path}"
        return f"{self.base_url} (key: {self.key_source(keys_file)})"


@dataclass
class Config:
    default_model: str | None = None
    default_provider: str | None = None
    temperature: float = 0.2
    max_steps: int = 25
    max_tool_output_chars: int = 24_000
    allow_outside_root: bool = False
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    keys: dict[str, str] = field(default_factory=dict)
    keys_permissions_loose: bool = False
    path: pathlib.Path | None = None

    # --------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: pathlib.Path | str | None = None) -> "Config":
        cfg = cls()
        p = pathlib.Path(path).expanduser() if path else config_path()
        cfg.path = p
        if p.exists():
            raw = tomllib.loads(p.read_text(encoding="utf-8"))
            cfg._apply_toml(raw)
        else:
            cfg._apply_implicit()
        kp = keys_path()
        if kp.exists():
            try:
                mode = kp.stat().st_mode & 0o777
                cfg.keys_permissions_loose = bool(mode & 0o077)
                cfg.keys.update(
                    {
                        str(k): str(v)
                        for k, v in tomllib.loads(kp.read_text(encoding="utf-8"))
                        .get("keys", {})
                        .items()
                    }
                )
            except (tomllib.TOMLDecodeError, OSError) as exc:
                raise ConfigError(f"cannot read {kp}: {exc}") from exc
        return cfg

    def _apply_toml(self, raw: dict[str, Any]) -> None:
        self.default_model = raw.get("default_model") or self.default_model
        self.default_provider = raw.get("default_provider") or self.default_provider
        self.temperature = float(raw.get("temperature", self.temperature))
        self.max_steps = int(raw.get("max_steps", self.max_steps))
        self.max_tool_output_chars = int(
            raw.get("max_tool_output_chars", self.max_tool_output_chars)
        )
        self.allow_outside_root = bool(raw.get("allow_outside_root", self.allow_outside_root))
        self.aliases = {str(k): str(v) for k, v in (raw.get("models") or {}).items()}
        for name, table in (raw.get("providers") or {}).items():
            self.providers[name] = self._provider_from(name, dict(table))
        self._apply_implicit()

    def _provider_from(self, name: str, table: dict[str, Any]) -> ProviderConfig:
        preset_name = table.pop("preset", name)
        merged = {**PRESETS.get(preset_name, {}), **table}
        return ProviderConfig(
            name=name,
            type=merged.get("type", "openai_compat"),
            base_url=merged.get("base_url"),
            api_key=merged.get("api_key"),
            api_key_env=merged.get("api_key_env", PRESETS.get(preset_name, {}).get("api_key_env")),
            model_path=merged.get("model_path"),
            default_model=merged.get("default_model")
            or PRESETS.get(preset_name, {}).get("default_model"),
            n_ctx=int(merged.get("n_ctx", 32768)),
            n_gpu_layers=int(merged.get("n_gpu_layers", -1)),
            n_threads=merged.get("n_threads"),
            chat_format=merged.get("chat_format"),
            tool_mode=merged.get("tool_mode", "auto"),
            extra={
                k: v
                for k, v in merged.items()
                if k
                not in {
                    "type",
                    "base_url",
                    "api_key",
                    "api_key_env",
                    "model_path",
                    "default_model",
                    "n_ctx",
                    "n_gpu_layers",
                    "n_threads",
                    "chat_format",
                    "tool_mode",
                    "preset",
                }
            },
        )

    def _apply_implicit(self) -> None:
        """Any known provider whose env var is set becomes usable with zero config."""
        for name in _KEYED_PRESETS:
            if name in self.providers:
                continue
            env = PRESETS[name].get("api_key_env")
            if env and os.environ.get(env):
                self.providers[name] = self._provider_from(name, {})

    # -------------------------------------------------------------- lookups
    def get_provider(self, name: str) -> ProviderConfig:
        if name in self.providers:
            return self.providers[name]
        if name in PRESETS:
            pc = self._provider_from(name, {})
            if pc.type == "gguf" or pc.resolve_key(self.keys):
                return pc
            raise ConfigError(
                f"provider '{name}' needs a credential: set ${PRESETS[name].get('api_key_env')} "
                f"or add [keys] {name} = \"...\" to {keys_path()}"
            )
        raise ConfigError(
            f"unknown provider '{name}'. Known: {', '.join(sorted(set(self.providers) | set(PRESETS)))}"
        )

    def resolve_model(self, ref: str | None = None) -> tuple[ProviderConfig, str]:
        """Turn ``"provider/model"``, an alias, or a bare model into a provider + model."""
        wanted = ref or self.default_model
        if wanted and wanted in self.aliases:
            wanted = self.aliases[wanted]
        if not wanted:
            for pc in self.providers.values():
                if pc.default_model and self._is_usable(pc):
                    return pc, pc.default_model
            if self.default_provider:
                pc = self.get_provider(self.default_provider)
                return pc, pc.default_model or "default"
            raise ConfigError(
                "no model selected. Pass --model provider/model, or set default_model in "
                f"{config_path()}, or run `openforge init`."
            )

        if "/" in wanted:
            head, tail = wanted.split("/", 1)
            if head in self.providers or head in PRESETS:
                pc = self.get_provider(head)
                return pc, tail
        # bare model name: use the default provider, or the only usable one
        if self.default_provider:
            return self.get_provider(self.default_provider), wanted
        usable = [pc for pc in self.providers.values() if self._is_usable(pc)]
        if len(usable) == 1:
            return usable[0], wanted
        raise ConfigError(
            f"cannot tell which provider serves model '{wanted}'. "
            f"Use 'provider/model' (e.g. openai/{wanted})."
        )

    def _is_usable(self, pc: ProviderConfig) -> bool:
        """A gguf provider is only usable when its weights exist on disk."""
        if pc.type == "gguf":
            try:
                return pc.resolved_model_path().exists()
            except ConfigError:
                return False
        return bool(pc.resolve_key(self.keys))

    def usable_providers(self) -> list[tuple[str, ProviderConfig, bool]]:
        out = []
        for name in sorted(set(self.providers) | set(PRESETS)):
            try:
                pc = self.get_provider(name)
            except ConfigError:
                preset = PRESETS.get(name, {})
                out.append(
                    (
                        name,
                        ProviderConfig(name=name, **{
                            k: preset[k]
                            for k in ("type", "base_url", "api_key_env", "default_model")
                            if k in preset
                        }),
                        False,
                    )
                )
                continue
            out.append((name, pc, self._is_usable(pc)))
        return out


CONFIG_TEMPLATE = """\
# OpenForge configuration — bring your own keys.
# Nothing here is sent anywhere except to the endpoint you configure.

# Which model `openforge` uses when you don't pass --model.
# Format: "provider/model", where provider is a [providers.<name>] table or a
# built-in preset (openai, anthropic, openrouter, groq, deepseek, together,
# mistral, xai, cerebras, fireworks, ollama, llama-server, lmstudio).
#
# If you leave this unset, OpenForge picks the first provider that has a key.
# default_model = "openai/gpt-4.1-mini"
# default_model = "local/qwen2.5-coder-7b"

# default_provider = "openai"   # used when you pass a bare model name
temperature = 0.2
max_steps = 25
allow_outside_root = false      # refuse to read/write outside the workspace

# Shortcuts usable as --model <alias>
[models]
# fast = "groq/llama-3.3-70b-versatile"
# smart = "anthropic/claude-sonnet-4-20250514"

# ---------------------------------------------------------------- providers
# Remote providers need nothing but a key in the environment:
#   export OPENAI_API_KEY=...        -> usable immediately as openai/<model>
#   export OPENROUTER_API_KEY=...    -> usable immediately as openrouter/<model>
# Add a table only when you want a custom endpoint, model or tool mode:

# [providers.work]
# preset = "openrouter"              # inherits base_url + OPENROUTER_API_KEY
# default_model = "anthropic/claude-sonnet-4"

# A self-hosted vLLM / gateway:
# [providers.vllm]
# type = "openai_compat"
# base_url = "http://gpu-box:8000/v1"
# api_key_env = "VLLM_API_KEY"

# Anthropic's native (non-OpenAI) API:
# [providers.anthropic]
# preset = "anthropic"

# An in-process GGUF model — nothing leaves the machine.
# Needs: pip install "openforge[gguf]"
# [providers.local]
# type = "gguf"
# model_path = "~/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
# n_ctx = 32768
# n_gpu_layers = -1                  # -1 = offload every layer (Metal/CUDA)
# chat_format = "chatml"             # only if auto-detection picks wrong
# tool_mode = "text"                 # "native" for templates with tool support
"""

KEYS_TEMPLATE = """\
# Secrets live here, never in config.toml and never in git.
# chmod 600 this file.  Alternatively just export the env vars.
[keys]
# openai = "sk-..."
# anthropic = "sk-ant-..."
# openrouter = "sk-or-v1-..."
# groq = "gsk_..."
"""
