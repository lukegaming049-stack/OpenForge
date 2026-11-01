"""Provider wire-format handling and config resolution."""

import json

import pytest

from openforge.config import PRESETS, Config, ConfigError
from openforge.providers.anthropic import to_anthropic
from openforge.providers.openai_compat import (
    finalize,
    iter_sse_json,
    merge_stream_delta,
    new_accumulator,
    to_openai_messages,
    to_openai_tools,
)
from openforge.tools import schemas


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for preset in PRESETS.values():
        env = preset.get("api_key_env")
        if env:
            monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("OPENFORGE_CONFIG_DIR", str(tmp_path / "cfg"))
    yield


# ---------------------------------------------------------------- SSE / deltas
def test_merge_stream_delta_accumulates_text_and_split_arguments():
    acc = new_accumulator()
    merge_stream_delta(acc, {"content": "Hel"})
    merge_stream_delta(acc, {"content": "lo"})
    merge_stream_delta(acc, {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read_file", "arguments": '{"pa'}}]})
    merge_stream_delta(acc, {"tool_calls": [{"index": 0, "function": {"arguments": 'th": "a.py"}'}}]})
    turn = finalize(acc)

    assert turn.text == "Hello"
    assert turn.tool_calls[0].name == "read_file"
    assert turn.tool_calls[0].arguments == {"path": "a.py"}
    assert turn.tool_calls[0].id == "c1"
    assert turn.finish_reason == "tool_calls"


def test_merge_stream_delta_does_not_duplicate_a_resent_name():
    acc = new_accumulator()
    merge_stream_delta(acc, {"tool_calls": [{"index": 0, "function": {"name": "list_dir"}}]})
    merge_stream_delta(acc, {"tool_calls": [{"index": 0, "function": {"name": "list_dir"}}]})
    assert finalize(acc).tool_calls[0].name == "list_dir"


def test_merge_handles_two_parallel_tool_calls():
    acc = new_accumulator()
    merge_stream_delta(acc, {"tool_calls": [{"index": 0, "function": {"name": "list_dir", "arguments": "{}"}}]})
    merge_stream_delta(acc, {"tool_calls": [{"index": 1, "function": {"name": "glob", "arguments": '{"pattern": "*"}'}}]})
    turn = finalize(acc)
    assert [c.name for c in turn.tool_calls] == ["list_dir", "glob"]


def test_finalize_skips_nameless_slots():
    acc = new_accumulator()
    merge_stream_delta(acc, {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]})
    assert finalize(acc).tool_calls == []


def test_iter_sse_json_skips_noise_and_stops_at_done():
    lines = [
        "",
        ": keep-alive",
        'data: {"choices":[{"delta":{"content":"a"}}]}',
        "data: [DONE]",
        'data: {"choices":[{"delta":{"content":"ignored"}}]}',
    ]
    payloads = list(iter_sse_json(lines))
    assert len(payloads) == 1
    assert payloads[0]["choices"][0]["delta"]["content"] == "a"


def test_iter_sse_json_survives_bad_json():
    assert list(iter_sse_json(["data: {oops", "data: {}"])) == [{}]


# ------------------------------------------------------------- wire conversion
def test_to_openai_messages_serialises_tool_calls():
    wire = to_openai_messages(
        [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "list_dir", "arguments": {"path": "."}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "list_dir", "content": "ok"},
            {"role": "assistant", "content": "hi", "tool_calls": []},
        ]
    )
    assert wire[0]["tool_calls"][0]["type"] == "function"
    assert json.loads(wire[0]["tool_calls"][0]["function"]["arguments"]) == {"path": "."}
    assert "tool_calls" not in wire[2]
    assert wire[1]["role"] == "tool"


def test_to_openai_tools_matches_registry():
    wire = to_openai_tools(schemas())
    names = [t["function"]["name"] for t in wire]
    assert "read_file" in names and "run_bash" in names
    assert wire[0]["type"] == "function"
    assert wire[0]["function"]["parameters"]["type"] == "object"


def test_anthropic_conversion_groups_tool_results_and_drops_leading_assistant():
    system, msgs = to_anthropic(
        [
            {"role": "system", "content": "be brief"},
            {"role": "assistant", "content": "orphan"},
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": "on it", "tool_calls": [{"id": "t1", "name": "list_dir", "arguments": {}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "a\nb"},
            {"role": "tool", "tool_call_id": "t2", "content": "c"},
        ]
    )
    assert system == "be brief"
    assert msgs[0]["role"] == "user"
    assert msgs[1]["content"][1] == {"type": "tool_use", "id": "t1", "name": "list_dir", "input": {}}
    assert [b["type"] for b in msgs[2]["content"]] == ["tool_result", "tool_result"]


# ---------------------------------------------------------------------- config
def test_resolve_model_with_provider_prefix(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = Config.load()
    pc, model = cfg.resolve_model("openai/gpt-4.1-mini")
    assert pc.name == "openai" and model == "gpt-4.1-mini"
    assert pc.base_url == PRESETS["openai"]["base_url"]


def test_model_names_containing_slashes(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    cfg = Config.load()
    pc, model = cfg.resolve_model("openrouter/meta-llama/llama-3.3-70b-instruct")
    assert pc.name == "openrouter"
    assert model == "meta-llama/llama-3.3-70b-instruct"


def test_missing_key_is_a_clear_error():
    cfg = Config.load()
    with pytest.raises(ConfigError) as exc:
        cfg.resolve_model("openai/gpt-4.1-mini")
    assert "needs a credential" in str(exc.value)


def test_alias_and_gguf_provider(tmp_path):
    model_file = tmp_path / "tiny.gguf"
    model_file.write_bytes(b"GGUF")
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        f"""
default_model = "local"
temperature = 0.1
[models]
cheap = "local/tiny"

[providers.local]
type = "gguf"
model_path = "{model_file}"
n_ctx = 4096
tool_mode = "text"
""",
        encoding="utf-8",
    )
    cfg = Config.load(cfg_file)
    pc, model = cfg.resolve_model("cheap")
    assert pc.type == "gguf" and pc.n_ctx == 4096 and pc.tool_mode == "text"
    assert model == "tiny"
    assert pc.resolved_model_path() == model_file


def test_preset_inheritance_with_overrides(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        """
[providers.work]
preset = "groq"
default_model = "llama-3.1-8b-instant"
""",
        encoding="utf-8",
    )
    cfg = Config.load(cfg_file)
    pc = cfg.providers["work"]
    assert pc.base_url == PRESETS["groq"]["base_url"]
    assert pc.api_key_env == "GROQ_API_KEY"
    assert pc.default_model == "llama-3.1-8b-instant"


def test_build_provider_requires_a_key_for_remote_hosts():
    from openforge.providers import build_provider
    from openforge.config import ProviderConfig

    with pytest.raises(ConfigError) as exc:
        build_provider(ProviderConfig(name="x", base_url="https://api.example.com/v1"))
    assert "no API key" in str(exc.value)


def test_build_provider_allows_keyless_localhost():
    from openforge.config import ProviderConfig
    from openforge.providers import build_provider

    provider = build_provider(
        ProviderConfig(name="llama", base_url="http://localhost:8080/v1"), model="local"
    )
    assert provider.name == "llama"
    assert provider.api_key is None


def test_keys_file_is_read_and_redacted(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENFORGE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "keys.toml").write_text('[keys]\nopenai = "sk-abcdef123456"\n', encoding="utf-8")
    cfg = Config.load()
    cfg.providers["openai"] = cfg._provider_from("openai", {})
    assert cfg.providers["openai"].resolve_key(cfg.keys) == "sk-abcdef123456"
    assert cfg.providers["openai"].key_source(cfg.keys) == "keys.toml"
