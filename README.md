# OpenForge(beta)

An open-source coding agent harness you run yourself: **bring your own key** (or
your own `.gguf`), a tool-calling loop that actually edits your repo, and a
terminal UI with a built-in editor so you can watch and correct the work as it
happens.

No accounts. No telemetry. No hosted proxy. The only network traffic is the
request you send to the endpoint you configured.

```
┌─ openforge ────────────────────────────────────────────────────────────────┐
│ file tree        │  Chat                          │  Editor                │
│ src/             │  you ▸ add a --json flag       │  src/cli.py       ●    │
│   cli.py         │  → grep  "argparse"            │   1 import argparse    │
│   agent.py       │  → read_file src/cli.py        │   2 ...                │
│ tests/           │  → edit_file src/cli.py        │                        │
│                  │    ┌── diff, approve? y/n ──┐  │                        │
│                  │    │ +    parser.add_arg…   │  │                        │
│                  │    └────────────────────────┘  │                        │
│                  │  Added --json, 1 file, tests…  │                        │
├──────────────────┴────────────────────────────────┴────────────────────────┤
│ openai:gpt-4.1-mini · tools:native · tok 1.2k↑ 340↓ · steps:4              │
│ Ask OpenForge…  (@file attaches, Enter sends)                              │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Why another one

Three things this does that most harnesses don't:

1. **BYOK only, and honest about it.** A provider is usable when, and only when,
   it can find a credential. No login, no seat management, no "free tier" that
   proxies your code through someone's server.
2. **Local GGUF, in-process.** Not an OpenAI-compatible server you have to launch
   first — `llama-cpp-python` loads the weights inside the agent, with GPU
   offload, KV-cache reuse, and a text tool protocol for the many chat templates
   that don't implement native tool calls.
3. **An editor in the same window.** The agent's writes land in the buffer you
   are looking at, and every write/shell command is a diff or a command you
   approve before it runs.

---

## Install

```bash
git clone <this repo> openforge && cd openforge
python -m venv .venv && source .venv/bin/activate
pip install -e .            # core: TUI + HTTP providers
pip install -e ".[gguf]"    # optional: in-process GGUF
```

Python 3.10+. Two entry points are installed: `openforge` and `forge`.

### First run

```bash
openforge init          # writes ~/.config/openforge/{config.toml,keys.toml}
openforge doctor        # checks python, textual, llama-cpp, keys, model, workspace
openforge               # opens the TUI
```

`init` never guesses at a model you don't have — it writes a commented template.

---

## Path 1 — a hosted model (30 seconds)

Any of these env vars is enough; no config file needed:

```bash
export OPENAI_API_KEY=sk-...          # or ANTHROPIC_API_KEY, OPENROUTER_API_KEY,
                                      # GROQ_API_KEY, DEEPSEEK_API_KEY, TOGETHER_API_KEY,
                                      # MISTRAL_API_KEY, XAI_API_KEY, CEREBRAS_API_KEY,
                                      # FIREWORKS_API_KEY
openforge run "why does tests/test_api.py::test_retry fail?"
```

Or put keys in `~/.config/openforge/keys.toml` (created `0600` by `init`):

```toml
[keys]
openai = "sk-..."
groq = "gsk_..."
```

Then name the model however you like:

```bash
openforge run "add type hints to utils.py" -m openai/gpt-4.1-mini
openforge -m groq/llama-3.3-70b-versatile          # TUI with that model
openforge -m openrouter/meta-llama/llama-3.3-70b-instruct
```

Local servers that need no key: `ollama/qwen2.5-coder:7b`,
`llama-server/local` (`llama-server -m model.gguf` on port 8080),
`lmstudio/local-model`.

---

## Path 2 — a local GGUF

```bash
pip install "openforge[gguf]"
```

GPU builds are the difference between 4 tok/s and 40:

```bash
# CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-binary :all:
# Apple Metal
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-binary :all:
```

Point a provider at the file:

```toml
[providers.local]
type = "gguf"
model_path = "~/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
n_ctx = 32768
n_gpu_layers = -1          # -1 = every layer on the GPU; use 0 for CPU-only
n_threads = 8              # CPU threads (omit to let llama.cpp decide)
tool_mode = "text"         # see below
```

```bash
openforge -m local/qwen2.5-coder-7b
```

**Model choice for agent work.** You want an instruct/coder model with a long
context and reliable instruction-following — 7B is the practical floor, 14–32B
is where agentic editing starts to feel dependable. Q4_K_M is the usual
quality/size sweet spot; drop to Q5_K_M or Q6_K if you have the RAM.

**`tool_mode`.** Most GGUF chat templates don't implement the OpenAI `tools`
contract, so the default here is `text`: the tool catalogue plus a strict
` ```tool {"name": ..., "arguments": {...}} ``` ` protocol go into the system
prompt, and the loop parses those blocks out. Set `tool_mode = "native"` for
templates that do support it (Hermes-2-Pro, Qwen2.5/Qwen3, Llama-3.1+ with a
recent llama-cpp-python) — it's more reliable when it works.

If you see the model narrating tool calls in prose instead of emitting blocks,
that's a template/prompt mismatch: try `native`, or a stronger model.

---

## The TUI

| Key | Action |
| --- | --- |
| `Enter` | send the prompt |
| `@path/to/file` | attach that file's contents to the message |
| `ctrl+s` | save the file in the editor |
| `ctrl+r` / `ctrl+l` | focus editor / prompt |
| `ctrl+n` | new chat (clears history) |
| `ctrl+q` | quit (saves the session) |
| `y` / `n` / `esc` | allow / deny the pending diff or command |

Slash commands: `/help`, `/open <path>`, `/tools`, `/model [ref]`, `/clear`,
`/yolo` (toggle auto-approve), `/save`, `/quit`.

The file tree opens files into the editor pane with syntax highlighting. When
the agent writes a file you have open, the buffer reloads — unless you have
unsaved changes, in which case you get a warning instead of losing your work.

---

## Tools

| Tool | What it does | Approval |
| --- | --- | --- |
| `read_file` | file contents with line numbers, paged | no |
| `write_file` | create or fully replace a file | **yes, with diff** |
| `edit_file` | exact-string replacement, unique-match enforced | **yes, with diff** |
| `list_dir` | directory tree, depth ≤ 4 | no |
| `glob` | find files by pattern | no |
| `grep` | regex content search, `path:line: text` | no |
| `run_bash` | shell command in the workspace, ≤ 300 s | **yes, command shown** |

`edit_file` fails loudly instead of guessing: not-found returns the closest
matching line, ambiguous matches list the line numbers and demand
`replace_all` or more context.

---

## CLI

```bash
openforge                                  # TUI in the cwd
openforge -C ~/code/api                    # TUI in another workspace
openforge run "..." -m local/qwen2.5-coder-7b
openforge run "..." --dry-run              # show diffs, write nothing
openforge run "..." -y                     # auto-approve (CI, sandboxes)
openforge run "..." --print-only -q        # just the final answer
openforge models                           # providers, endpoints, key status
openforge doctor                           # diagnose the install
openforge config                           # resolved config, keys redacted
```

Exit codes: `0` ok, `1` runtime error, `2` config error, `130` interrupted.

---

## Configuration reference

`~/.config/openforge/config.toml` (override with `--config` or
`OPENFORGE_CONFIG_DIR`):

| Key | Default | Meaning |
| --- | --- | --- |
| `default_model` | unset | `provider/model`, or an alias from `[models]` |
| `default_provider` | unset | provider for bare model names |
| `temperature` | `0.2` | sampling temperature |
| `max_steps` | `25` | tool round-trips before the run is stopped |
| `max_tool_output_chars` | `24000` | per-tool-result truncation |
| `allow_outside_root` | `false` | allow paths outside the workspace |
| `[models]` | — | aliases: `fast = "groq/llama-3.3-70b-versatile"` |

`[providers.<name>]` fields: `type` (`openai_compat` \| `anthropic` \| `gguf`),
`preset`, `base_url`, `api_key`, `api_key_env`, `default_model`, `tool_mode`
(`auto` \| `native` \| `text`), and for GGUF `model_path`, `n_ctx`,
`n_gpu_layers`, `n_threads`, `chat_format`. Unknown keys are passed through as
extra request body fields (`top_p`, `n_batch`, `flash_attn`, …).

**Key resolution order:** `keys.toml` → `api_key` in `config.toml` →
`$<api_key_env>` → missing (provider unusable). `openforge config` prints which
one won, redacted.

---

## How it fits together

```
cli.py ──┬─ tui/app.py ── editor.py          (Textual: tree, chat, editor)
         └─ run (headless)
              │
              ├─ config.py       BYOK resolution, presets, model refs
              ├─ agent.py        the loop: prompt → model → tools → … → answer
              │    ├─ prompts.py         system prompt + text tool protocol
              │    └─ tools/             read/write/edit/list/glob/grep/bash
              │         └─ context.py    path guard + approval requests
              └─ providers/
                   ├─ openai_compat.py   OpenAI, OpenRouter, Groq, vLLM, Ollama…
                   ├─ anthropic.py       Messages API, native tool use
                   └─ gguf.py            llama-cpp-python, in-process
```

One internal message format (OpenAI-ish dicts), one `Turn` return type, one list
of `ToolCall`s — so the loop is identical whether the answer came from a
frontier API or a 7B on your laptop.

The loop: append the prompt → call the model (streaming) → if it asked for
tools, execute each one, append results, call again → stop when it answers in
prose. Guards: identical repeated calls abort after 3 tries, `max_steps` caps the
run, and old tool output is elided once the transcript passes ~400k characters.

---

## Security model

- **Path guard.** Every tool path resolves against the workspace root and is
  refused if it escapes, unless `allow_outside_root = true`. `../../etc/passwd`
  is a tool error, not a read.
- **Approval gate.** Every write shows a unified diff; every shell command shows
  the command and cwd. Denials are reported back to the model as a normal tool
  result so it changes approach instead of retrying.
- **Keys** live in the environment or in `keys.toml` with mode `0600` (`doctor`
  checks it). They are never written to config, sessions or logs, and are
  redacted everywhere they're printed.
- **No telemetry, no analytics, no update pings.** `run_bash` executes with your
  user's privileges — that is the point of the tool, and the reason approval
  defaults to on.

Auto-approve (`-y`, `/yolo`) removes the human check. Use it in containers and
throwaway checkouts, not on a repo you care about.

---

## Extending

**A new provider** is ~15 lines if it speaks OpenAI's wire format:

```toml
[providers.mygateway]
type = "openai_compat"
base_url = "https://gateway.internal/v1"
api_key_env = "MYGATEWAY_KEY"
default_model = "some-model"
```

A genuinely different protocol: subclass `Provider`, implement `chat()`, return
`Turn(text=..., tool_calls=[ToolCall(...)])`, and register it in
`providers/__init__.py:build_provider`.

**A new tool** is one registry entry plus one function:

```python
# openforge/tools/files.py
def file_size(ctx: ToolContext, path: str, **_) -> str:
    target = ctx.resolve(path, must_exist=True)
    return f"{target.stat().st_size} bytes"
```

```python
# openforge/tools/__init__.py
"file_size": Tool(
    name="file_size",
    description="Byte size of a file.",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    run=files.file_size,
),
```

It appears in the system prompt, in the native schemas, and in the dispatcher
automatically. Set `writes=True` if it mutates the workspace.

---

## Tests

```bash
pip install -e ".[dev]"
pytest -q          # 59 tests: tools, loop, wire formats, config, HTTP, TUI
```

The interesting ones: `tests/test_integration_http.py` runs a real loopback HTTP
server that streams SSE tool calls, so the provider → parser → tool → file path
is exercised end to end without a vendor; `tests/test_tui_agent.py` drives the
actual TUI headlessly and approves a diff through the modal.

---

## Known limitations

- **No MCP, no sub-agents, no multi-file planning.** This is a single-agent loop
  with seven tools. Deliberately.
- **Text tool protocol is lossy** with small models: they sometimes emit prose
  instead of a block, or malformed JSON (which comes back as a readable error so
  they can retry).
- **No repo indexing.** `grep`/`glob` are literal; on large repos expect several
  round-trips to find things.
- **Anthropic streaming isn't token-by-token** — its provider returns the whole
  turn at once. Functionally identical, less pretty.
- **GGUF tool calling depends on the template.** Native mode is only as good as
  the chat template baked into the file.

## License

MIT.
