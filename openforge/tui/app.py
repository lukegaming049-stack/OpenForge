"""The TUI: file tree, chat pane, built-in editor, approval modals.

The agent runs in a worker thread; every UI mutation is marshalled back to the
main thread with ``call_from_thread`` so Textual's event loop stays honest.
"""

from __future__ import annotations

import json
import pathlib
import threading
import time
from typing import Any

from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    DirectoryTree,
    Footer,
    Header,
    Input,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)

from .. import session
from ..agent import Agent, AgentEvent
from ..config import Config, ConfigError
from ..providers import build_provider
from ..tools import ToolContext, tool_names
from .editor import CodeEditor

HELP_TEXT = """\
Prompt  Enter sends.  `@path/to/file` attaches that file to your message.
Keys    ctrl+s save file · ctrl+r editor · ctrl+l prompt · ctrl+n new chat · ctrl+q quit
Slash   /help /open <path> /tools /model [ref] /clear /yolo /save /quit
Approve every write shows a diff; every shell command shows the command. `y` allows, `n` denies.
"""


def _post_safe(app: "ForgeApp", fn: Any, *args: Any) -> None:
    if threading.current_thread() is threading.main_thread():
        fn(*args)
    else:
        app.call_from_thread(fn, *args)


class ConfirmScreen(ModalScreen[bool]):
    """Approval dialog. Returns True to allow, False to deny."""

    BINDINGS = [
        Binding("y", "allow", "Allow"),
        Binding("n", "deny", "Deny"),
        Binding("escape", "deny", "Deny"),
    ]

    def __init__(self, request: Any) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Static(Text(self.request.title, style="bold"), id="confirm-title")
            with VerticalScroll(id="confirm-detail"):
                yield Static(self._detail_renderable())
            yield Static("y allow      n deny", id="confirm-hint")

    def _detail_renderable(self) -> Any:
        detail = self.request.detail or ""
        try:
            if self.request.kind == "write":
                return Syntax(detail, "diff", theme="monokai", word_wrap=False)
            return Syntax(detail, "bash", theme="monokai", word_wrap=True)
        except Exception:  # unknown lexer / theme: plain text is fine
            return Text(detail)

    def action_allow(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)


class ForgeApp(App[None]):
    """OpenForge's terminal UI."""

    CSS = """
    Screen { layout: vertical; }
    #body { height: 1fr; }
    #tree { width: 34; border-right: solid $panel; }
    #right { width: 1fr; }
    #tabs { height: 1fr; }
    #log { height: 1fr; padding: 0 1; }
    #stream { height: auto; max-height: 12; padding: 0 1; color: $text-muted; }
    #status { height: 1; padding: 0 1; background: $panel; color: $text-muted; }
    #prompt { dock: bottom; }
    CodeEditor { height: 1fr; }
    #editor-title { height: 1; padding: 0 1; background: $boost; }
    #editor-body { height: 1fr; }
    ConfirmScreen { align: center middle; }
    #confirm-box {
        width: 92%; max-width: 140; height: 82%;
        border: thick $accent; background: $surface; padding: 1 2;
    }
    #confirm-detail { height: 1fr; }
    #confirm-hint { height: 1; color: $text-muted; }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+s", "save_file", "Save"),
        Binding("ctrl+r", "focus_editor", "Editor"),
        Binding("ctrl+l", "focus_prompt", "Prompt"),
        Binding("ctrl+n", "new_chat", "New chat"),
    ]

    def __init__(
        self,
        config: Config,
        model_ref: str | None = None,
        root: pathlib.Path | str | None = None,
        *,
        yolo: bool = False,
        resume: bool = False,
    ) -> None:
        super().__init__()
        self.config_obj = config
        self.model_ref = model_ref
        self.root = pathlib.Path(root or pathlib.Path.cwd()).expanduser().resolve()
        self.yolo = yolo
        self.resume = resume
        self.agent: Agent | None = None
        self._agent_lock = threading.Lock()
        self._stream_buf = ""
        self._last_flush = 0.0
        self._busy = False

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            yield DirectoryTree(str(self.root), id="tree")
            with Vertical(id="right"):
                with TabbedContent(initial="tab-chat", id="tabs"):
                    with TabPane("Chat", id="tab-chat"):
                        yield RichLog(id="log", wrap=True, markup=False, highlight=True)
                        yield Static("", id="stream")
                    with TabPane("Editor", id="tab-editor"):
                        yield CodeEditor(self.root, id="editor")
                yield Static("", id="status")
        yield Input(placeholder="Ask OpenForge…  (@file attaches, Enter sends)", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "OpenForge"
        self.sub_title = str(self.root)
        self.log_widget.write(Text(f"OpenForge — {self.root}", style="bold"))
        try:
            pc, model = self.config_obj.resolve_model(self.model_ref)
            self.log_widget.write(Text(f"model: {pc.name}/{model}  (tools: {pc.tool_mode})", style="dim"))
        except ConfigError as exc:
            self.log_widget.write(Text(f"model not resolved: {exc}", style="yellow"))
        self.log_widget.write(Text("Type /help for keys and commands. Nothing leaves this machine except API calls to your provider.", style="dim"))
        self._update_status()
        if self.resume:
            data = session.load()
            if data and data.get("messages"):
                self.log_widget.write(Text(f"resumed {len(data['messages'])} messages from {session.last_session_path()}", style="dim"))
        self.query_one("#prompt", Input).focus()

    # --------------------------------------------------------------- shortcuts
    @property
    def log_widget(self) -> RichLog:
        return self.query_one("#log", RichLog)

    @property
    def stream_widget(self) -> Static:
        return self.query_one("#stream", Static)

    @property
    def editor(self) -> CodeEditor:
        return self.query_one("#editor", CodeEditor)

    @property
    def prompt_input(self) -> Input:
        return self.query_one("#prompt", Input)

    def action_save_file(self) -> None:
        self.editor.save()

    def action_focus_editor(self) -> None:
        self.query_one("#tabs", TabbedContent).active = "tab-editor"
        self.editor.body.focus()

    def action_focus_prompt(self) -> None:
        self.prompt_input.focus()

    def action_new_chat(self) -> None:
        if self.agent:
            self.agent.reset()
        self._stream_buf = ""
        self.stream_widget.update("")
        self.log_widget.clear()
        self.log_widget.write(Text("new chat", style="dim"))
        self._update_status()

    def action_quit(self) -> None:
        self._persist()
        self.exit()

    # ------------------------------------------------------------------ events
    @on(DirectoryTree.FileSelected)
    def _open_from_tree(self, event: DirectoryTree.FileSelected) -> None:
        if self.editor.open_path(pathlib.Path(event.path)):
            self.query_one("#tabs", TabbedContent).active = "tab-editor"
            self.editor.body.focus()

    @on(Input.Submitted, "#prompt")
    def _on_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        self.prompt_input.value = ""
        if not text:
            return
        if text.startswith("/"):
            self._command(text)
            return
        self.log_widget.write(Text(f"you ▸ {text}", style="bold"))
        self._run_agent(text)

    # ---------------------------------------------------------------- commands
    def _command(self, raw: str) -> None:
        parts = raw[1:].split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("help", "?"):
            self.log_widget.write(Text(HELP_TEXT, style="dim"))
        elif cmd in ("quit", "q", "exit"):
            self.action_quit()
        elif cmd == "clear":
            self.action_new_chat()
        elif cmd == "tools":
            self.log_widget.write(Text("tools: " + ", ".join(tool_names()), style="dim"))
        elif cmd == "open":
            if not arg:
                self.log_widget.write(Text("usage: /open path/to/file", style="yellow"))
            elif self.editor.open_path(arg):
                self.action_focus_editor()
        elif cmd == "save":
            self.editor.save()
        elif cmd == "yolo":
            self.yolo = not self.yolo
            self.log_widget.write(
                Text(
                    "auto-approve ON — writes and commands run without asking" if self.yolo
                    else "auto-approve OFF — every write and command asks",
                    style="yellow" if self.yolo else "dim",
                )
            )
        elif cmd == "model":
            if not arg:
                self.log_widget.write(Text(f"current: {self.model_ref or self.config_obj.default_model}", style="dim"))
                return
            self.model_ref = arg
            if self.agent:
                self.agent.provider.close()
            self.agent = None
            self.log_widget.write(Text(f"model set to {arg}", style="dim"))
            self._update_status()
        else:
            self.log_widget.write(Text(f"unknown command /{cmd} — try /help", style="yellow"))

    # ------------------------------------------------------------------- agent
    def _ensure_agent(self) -> Agent:
        with self._agent_lock:
            if self.agent is not None:
                return self.agent
            pc, model = self.config_obj.resolve_model(self.model_ref)
            provider = build_provider(pc, model, self.config_obj.keys)
            ctx = ToolContext(
                root=self.root,
                approve=self._approve,
                allow_outside_root=self.config_obj.allow_outside_root,
                on_change=lambda path: _post_safe(self, self._on_file_changed, path),
            )
            history = None
            if self.resume:
                data = session.load()
                history = (data or {}).get("messages")
            self.agent = Agent(
                provider,
                model,
                ctx,
                config=self.config_obj,
                tool_mode=pc.tool_mode,
                on_event=lambda ev: _post_safe(self, self._apply_event, ev),
                history=history,
            )
            return self.agent

    def _approve(self, request: Any) -> bool:
        """Called from the worker thread: block until the modal answers."""
        if self.yolo:
            return True
        answer: dict[str, bool] = {}
        done = threading.Event()

        def show() -> None:
            def finished(result: bool | None) -> None:
                answer["ok"] = bool(result)
                done.set()

            try:
                self.push_screen(ConfirmScreen(request), callback=finished)
            except Exception:
                answer["ok"] = False
                done.set()

        try:
            self.call_from_thread(show)
        except Exception:
            return False
        done.wait(timeout=900)
        return bool(answer.get("ok"))

    @work(thread=True, exclusive=True, group="agent")
    def _run_agent(self, prompt: str) -> None:
        _post_safe(self, self._set_busy, True)
        try:
            agent = self._ensure_agent()
            _post_safe(self, self._set_status, f"{agent.provider.name}:{agent.model} · working…")
            agent.run(prompt)
        except ConfigError as exc:
            _post_safe(self, self._show_error, str(exc))
        except Exception as exc:  # provider/network/tool failures land here
            _post_safe(self, self._show_error, f"{type(exc).__name__}: {exc}")
        finally:
            _post_safe(self, self._set_busy, False)
            _post_safe(self, self._update_status)

    # ------------------------------------------------------------ agent events
    def _apply_event(self, event: AgentEvent) -> None:
        if event.kind == "delta":
            self._stream_buf += event.text
            now = time.monotonic()
            if now - self._last_flush >= 0.06:
                self._last_flush = now
                self._flush_stream()
        elif event.kind == "assistant":
            self._stream_buf = ""
            self.stream_widget.update("")
            body = event.text.strip()
            try:
                self.log_widget.write(Markdown(body) if body else Text("(no output)", style="dim"))
            except Exception:
                self.log_widget.write(Text(body))
        elif event.kind == "tool_start":
            name = event.data.get("name", "?")
            args = json.dumps(event.data.get("arguments") or {}, ensure_ascii=False)
            self.log_widget.write(Text(f"→ {name} {args[:240]}", style="cyan"))
        elif event.kind == "tool_result":
            text = event.text or ""
            if len(text) > 1500:
                text = text[:1500] + "\n… (truncated in view)"
            style = "red" if event.data.get("is_error") else "grey62"
            self.log_widget.write(Text(text, style=style))
        elif event.kind == "notice":
            self.log_widget.write(Text(f"· {event.text}", style="yellow"))
        elif event.kind == "error":
            self.log_widget.write(Text(f"! {event.text}", style="bold red"))
        elif event.kind == "done":
            self._update_status()

    def _flush_stream(self) -> None:
        tail = self._stream_buf[-3000:]
        self.stream_widget.update(Text(tail, style="italic"))

    def _show_error(self, message: str) -> None:
        self.log_widget.write(Text(f"! {message}", style="bold red"))

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.prompt_input.disabled = busy
        if not busy:
            self._flush_stream()
            self.prompt_input.focus()

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(Text(text, style="dim"))

    def _update_status(self) -> None:
        bits = []
        if self.agent:
            bits.append(f"{self.agent.provider.name}:{self.agent.model}")
            bits.append(f"tools:{self.agent.tool_mode}")
            used = self.agent.usage
            if used["prompt_tokens"] or used["completion_tokens"]:
                bits.append(f"tok {used['prompt_tokens']}↑ {used['completion_tokens']}↓")
            bits.append(f"steps:{self.agent.steps_taken}")
        else:
            bits.append(str(self.model_ref or self.config_obj.default_model or "no model"))
        if self.yolo:
            bits.append("AUTO-APPROVE")
        self._set_status("  ·  ".join(bits))

    # --------------------------------------------------------------- file sync
    def _on_file_changed(self, path: pathlib.Path) -> None:
        self.editor.refresh_if_open(path)
        try:
            self.query_one("#tree", DirectoryTree).reload()
        except Exception:
            pass

    # ----------------------------------------------------------------- persist
    def _persist(self) -> None:
        if not self.agent or not self.agent.messages:
            return
        try:
            session.save(self.agent.export())
        except OSError:
            pass

    def on_unmount(self) -> None:
        self._persist()
        if self.agent:
            self.agent.provider.close()
