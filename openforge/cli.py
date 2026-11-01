"""Command line entry point: ``openforge`` (aka ``forge``)."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

from . import __version__, session
from .config import (
    CONFIG_TEMPLATE,
    KEYS_TEMPLATE,
    Config,
    ConfigError,
    config_dir,
    config_path,
    keys_path,
    redact,
)


# --------------------------------------------------------------------- helpers
def _console():
    from rich.console import Console

    return Console(highlight=False, soft_wrap=False)


def _approval(auto_yes: bool, console) -> "callable":
    """Interactive approve/deny for headless runs."""
    from rich.text import Text

    state = {"all": auto_yes}

    def approve(request) -> bool:
        if state["all"]:
            console.print(Text(f"[auto-approved] {request.title}", style="yellow"))
            return True
        console.rule(Text(request.title, style="bold"))
        if request.detail:
            console.print(_detail_renderable(request))
        console.rule()
        while True:
            try:
                answer = input("allow? [y]es / [n]o / [a]ll / [q]uit: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return False
            if answer in ("y", "yes"):
                return True
            if answer in ("a", "all"):
                state["all"] = True
                return True
            if answer in ("n", "no", ""):
                return False
            if answer in ("q", "quit"):
                raise SystemExit(130)

    return approve


def _detail_renderable(request):
    from rich.syntax import Syntax
    from rich.text import Text

    try:
        lexer = "diff" if request.kind == "write" else "bash"
        return Syntax(request.detail, lexer, theme="monokai", word_wrap=False)
    except Exception:
        return Text(request.detail or "")


def _resolve_root(value: str | None) -> pathlib.Path:
    return pathlib.Path(value or os.getcwd()).expanduser().resolve()


# ------------------------------------------------------------------- commands
def cmd_run(args: argparse.Namespace) -> int:
    from rich.text import Text

    from .agent import Agent
    from .providers import build_provider
    from .tools import ToolContext

    console = _console()
    try:
        cfg = Config.load(args.config)
        pc, model = cfg.resolve_model(args.model)
        provider = build_provider(pc, model, cfg.keys)
    except ConfigError as exc:
        console.print(Text(f"config error: {exc}", style="bold red"))
        return 2

    root = _resolve_root(args.root)
    if not root.is_dir():
        console.print(Text(f"workspace not found: {root}", style="bold red"))
        return 2

    quiet = args.quiet or args.print_only

    def on_event(event) -> None:
        if quiet:
            return
        if event.kind == "delta":
            console.print(event.text, end="", markup=False, highlight=False)
        elif event.kind == "assistant":
            console.print()
        elif event.kind == "tool_start":
            name = event.data.get("name")
            console.print(Text(f"\n→ {name}", style="cyan"))
        elif event.kind == "tool_result":
            text = event.text or ""
            if len(text) > 600:
                text = text[:600] + " …"
            console.print(
                Text(text, style="red" if event.data.get("is_error") else "grey62")
            )
        elif event.kind == "notice":
            console.print(Text(f"· {event.text}", style="yellow"))
        elif event.kind == "error":
            console.print(Text(f"\n! {event.text}", style="bold red"))

    ctx = ToolContext(
        root=root,
        approve=_approval(args.yes, console),
        allow_outside_root=cfg.allow_outside_root,
        dry_run=args.dry_run,
    )
    history = None
    if args.resume:
        data = session.load(args.session)
        history = (data or {}).get("messages")
    agent = Agent(
        provider,
        model,
        ctx,
        config=cfg,
        tool_mode=pc.tool_mode,
        on_event=on_event,
        history=history,
    )
    try:
        answer = agent.run(args.prompt)
    except KeyboardInterrupt:
        console.print(Text("\ninterrupted", style="yellow"))
        return 130
    except Exception as exc:
        console.print(Text(f"\nerror: {exc}", style="bold red"))
        return 1
    finally:
        if not quiet:
            console.print()
        try:
            session.save(agent.export(), args.session)
        except OSError:
            pass
        provider.close()

    if args.print_only:
        print(answer)
    return 0


def cmd_tui(args: argparse.Namespace) -> int:
    from .tui.app import ForgeApp

    try:
        cfg = Config.load(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    app = ForgeApp(
        cfg,
        model_ref=args.model,
        root=_resolve_root(args.root),
        yolo=args.yes,
        resume=args.resume,
    )
    app.run()
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from rich.table import Table

    console = _console()
    cfg = Config.load(args.config)
    table = Table(title="providers", show_lines=False)
    table.add_column("name", style="bold")
    table.add_column("type")
    table.add_column("endpoint / model file")
    table.add_column("key")
    table.add_column("ready")
    for name, pc, ok in cfg.usable_providers():
        endpoint = pc.model_path if pc.type == "gguf" else (pc.base_url or "-")
        table.add_row(
            name,
            pc.type,
            str(endpoint),
            pc.key_source(cfg.keys),
            "[green]yes[/green]" if ok else "[red]no[/red]",
        )
    console.print(table)
    if cfg.aliases:
        console.print("\n[bold]aliases[/bold]")
        for alias, target in sorted(cfg.aliases.items()):
            console.print(f"  {alias} → {target}")
    console.print(f"\ndefault model: [bold]{cfg.default_model or '(unset)'}[/bold]")
    console.print(f"config: {cfg.path}")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    console = _console()
    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    cfg_file = config_path()
    keys_file = keys_path()

    if cfg_file.exists() and not args.force:
        console.print(f"config already exists: {cfg_file} (use --force to overwrite)")
    else:
        cfg_file.write_text(CONFIG_TEMPLATE, encoding="utf-8")
        console.print(f"wrote {cfg_file}")

    if keys_file.exists() and not args.force:
        console.print(f"keys file already exists: {keys_file}")
    else:
        keys_file.write_text(KEYS_TEMPLATE, encoding="utf-8")
        os.chmod(keys_file, 0o600)
        console.print(f"wrote {keys_file} (mode 600)")

    console.print(
        "\nNext:\n"
        "  1. put a key in the keys file or export one, e.g. export OPENAI_API_KEY=...\n"
        "  2. set default_model = \"openai/gpt-4.1-mini\" (or a gguf provider) in config.toml\n"
        "  3. run `openforge doctor`, then `openforge`"
    )
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from rich.markup import escape
    from rich.table import Table

    console = _console()
    table = Table(title="openforge doctor")
    table.add_column("check")
    table.add_column("result")
    table.add_column("detail")

    def row(name: str, ok: bool, detail: str = "") -> None:
        table.add_row(
            name,
            "[green]ok[/green]" if ok else "[red]fail[/red]",
            detail,
        )

    row("python", sys.version_info >= (3, 10), sys.version.split()[0])
    try:
        import textual  # noqa: F401

        row("textual (TUI)", True, getattr(textual, "__version__", "?"))
    except ImportError:
        row("textual (TUI)", False, "pip install textual")

    try:
        import llama_cpp  # noqa: F401

        row("llama-cpp-python (GGUF)", True, getattr(llama_cpp, "__version__", "?"))
    except ImportError:
        row(
            "llama-cpp-python (GGUF)",
            False,
            escape("optional — pip install 'openforge[gguf]'"),
        )

    cfg_file = config_path()
    row("config file", cfg_file.exists(), str(cfg_file))
    if keys_path().exists():
        mode = keys_path().stat().st_mode & 0o777
        row("keys file perms", mode & 0o077 == 0, f"{oct(mode)} — should be 0o600")
    else:
        row("keys file", True, "not used (env vars only)")

    try:
        cfg = Config.load(args.config)
    except ConfigError as exc:
        row("config parse", False, str(exc))
        console.print(table)
        return 1

    ready = [n for n, _pc, ok in cfg.usable_providers() if ok]
    row("usable providers", bool(ready), ", ".join(ready) or "none — set a key")

    try:
        pc, model = cfg.resolve_model(args.model)
        row("default model", True, f"{pc.name}/{model} (tools: {pc.tool_mode})")
        if pc.type == "gguf":
            path = pc.resolved_model_path()
            row("gguf weights", path.exists(), str(path))
    except ConfigError as exc:
        row("default model", False, str(exc))

    root = _resolve_root(args.root)
    row("workspace", root.is_dir() and os.access(root, os.W_OK), str(root))

    console.print(table)
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    console = _console()
    try:
        cfg = Config.load(args.config)
    except ConfigError as exc:
        console.print(f"config error: {exc}")
        return 2
    console.print(f"config: {config_path()}")
    console.print(f"keys:   {keys_path()}")
    console.print(f"default_model: {cfg.default_model}")
    console.print(f"temperature: {cfg.temperature}   max_steps: {cfg.max_steps}")
    console.print(f"allow_outside_root: {cfg.allow_outside_root}")
    for name, pc in sorted(cfg.providers.items()):
        key = pc.resolve_key(cfg.keys)
        console.print(f"  [{name}] {pc.type} {pc.base_url or pc.model_path or ''} key={redact(key)}")
    return 0


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openforge",
        description="OpenForge — a BYOK coding agent with a built-in editor and local GGUF support.",
    )
    parser.add_argument("--version", action="version", version=f"openforge {__version__}")
    parser.add_argument("--config", help="path to config.toml (default: ~/.config/openforge/config.toml)")
    sub = parser.add_subparsers(dest="command")

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("-m", "--model", help="provider/model, or an alias from [models]")
        p.add_argument("-C", "--root", help="workspace directory (default: cwd)")

    p_tui = sub.add_parser("tui", help="open the terminal UI (default)")
    common(p_tui)
    p_tui.add_argument("-y", "--yes", action="store_true", help="auto-approve every write and command")
    p_tui.add_argument("--resume", action="store_true", help="restore the last session's history")
    p_tui.set_defaults(func=cmd_tui)

    p_run = sub.add_parser("run", help="one-shot headless run")
    p_run.add_argument("prompt", help="what you want done")
    common(p_run)
    p_run.add_argument("-y", "--yes", action="store_true", help="auto-approve every write and command")
    p_run.add_argument("--dry-run", action="store_true", help="show what would change, write nothing")
    p_run.add_argument("--resume", action="store_true", help="continue the last session")
    p_run.add_argument("--session", help="session json path")
    p_run.add_argument("--print-only", action="store_true", help="print only the final answer")
    p_run.add_argument("-q", "--quiet", action="store_true", help="no streaming, no tool chatter")
    p_run.set_defaults(func=cmd_run)

    p_models = sub.add_parser("models", help="list providers and their key status")
    p_models.set_defaults(func=cmd_models)

    p_init = sub.add_parser("init", help="write a starter config.toml and keys.toml")
    p_init.add_argument("--force", action="store_true", help="overwrite existing files")
    p_init.set_defaults(func=cmd_init)

    p_doc = sub.add_parser("doctor", help="check the install, keys, model and workspace")
    common(p_doc)
    p_doc.set_defaults(func=cmd_doctor)

    p_cfg = sub.add_parser("config", help="show the resolved configuration (keys redacted)")
    p_cfg.set_defaults(func=cmd_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        # bare `openforge` opens the TUI
        args = parser.parse_args(["tui", *argv])
    try:
        return int(args.func(args) or 0)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
