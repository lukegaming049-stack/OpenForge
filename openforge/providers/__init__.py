"""Provider factory: config -> a ready-to-chat Provider."""

from __future__ import annotations

from urllib.parse import urlparse

from ..config import Config, ConfigError, ProviderConfig
from .base import Provider, ProviderError, ToolCall, Turn

__all__ = [
    "Provider",
    "ProviderError",
    "ToolCall",
    "Turn",
    "build_provider",
    "LOCAL_HOSTS",
]

LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal"}


def _is_local(base_url: str | None) -> bool:
    if not base_url:
        return False
    host = urlparse(base_url).hostname or ""
    return host in LOCAL_HOSTS or host.startswith("192.168.") or host.startswith("10.")


def build_provider(
    cfg: ProviderConfig, model: str | None = None, keys: dict[str, str] | None = None
) -> Provider:
    """Instantiate the provider described by ``cfg``."""
    chosen_model = model or cfg.default_model or "default"

    if cfg.type == "gguf":
        from .gguf import GGUFProvider

        path = cfg.resolved_model_path()
        if not path.exists():
            raise ConfigError(
                f"GGUF model not found: {path}\n"
                "Download a .gguf (e.g. from Hugging Face) or fix model_path in config.toml."
            )
        return GGUFProvider(
            name=cfg.name,
            model=chosen_model,
            model_path=str(path),
            n_ctx=cfg.n_ctx,
            n_gpu_layers=cfg.n_gpu_layers,
            n_threads=cfg.n_threads,
            chat_format=cfg.chat_format,
            tool_mode=cfg.tool_mode,
            extra={
                k: v
                for k, v in cfg.extra.items()
                if k in {"n_batch", "n_ubatch", "rope_freq_base", "rope_freq_scale", "flash_attn"}
            },
        )

    key = cfg.resolve_key(keys)

    if cfg.type == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider(
            name=cfg.name,
            model=chosen_model,
            api_key=key,
            base_url=cfg.base_url or "https://api.anthropic.com",
        )

    if cfg.type not in ("openai_compat", "openai", "local"):
        raise ConfigError(f"provider '{cfg.name}': unknown type '{cfg.type}'")
    if not cfg.base_url:
        raise ConfigError(f"provider '{cfg.name}': type=openai_compat needs a base_url")
    if not key and not _is_local(cfg.base_url):
        raise ConfigError(
            f"provider '{cfg.name}' has no API key. Export ${cfg.api_key_env or 'YOUR_KEY'} "
            f"or add to ~/.config/openforge/keys.toml:\n  [keys]\n  {cfg.name} = \"sk-...\""
        )

    from .openai_compat import OpenAICompatProvider

    extra_headers = {}
    extra_body = {}
    for k, v in cfg.extra.items():
        if k.startswith("header_"):
            extra_headers[k[len("header_") :].replace("_", "-")] = str(v)
        elif k not in {"timeout"}:
            extra_body[k] = v

    return OpenAICompatProvider(
        name=cfg.name,
        model=chosen_model,
        base_url=cfg.base_url,
        api_key=key,
        extra_headers=extra_headers,
        extra_body=extra_body,
        timeout=float(cfg.extra.get("timeout", 600)),
    )


def provider_for_model(cfg: Config, ref: str | None = None) -> tuple[Provider, str]:
    """Convenience: resolve a model reference and build its provider."""
    pc, model = cfg.resolve_model(ref)
    return build_provider(pc, model, cfg.keys), model
