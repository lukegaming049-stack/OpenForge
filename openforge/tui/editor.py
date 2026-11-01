"""Built-in editor pane.

A TextArea with syntax highlighting, dirty tracking and Ctrl+S.  The agent's
writes are pushed back into the open buffer, so the file you are looking at is
always the file on disk.
"""

from __future__ import annotations

import pathlib

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Static, TextArea

LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".json": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "css",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".sql": "sql",
    ".xml": "xml",
    ".ini": "ini",
    ".cfg": "ini",
    ".dockerfile": "dockerfile",
}

THEME = "monokai"


def language_for(path: pathlib.Path) -> str | None:
    if path.name.lower() in {"dockerfile", "makefile"}:
        return "dockerfile" if path.name.lower() == "dockerfile" else "makefile"
    return LANGUAGE_BY_SUFFIX.get(path.suffix.lower())


class CodeEditor(Vertical):
    """A minimal file editor. Ctrl+S writes; the title shows the dirty state."""

    BINDINGS = [Binding("ctrl+s", "save", "Save", show=True)]

    def __init__(self, root: pathlib.Path, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.root = root
        self.current_path: pathlib.Path | None = None
        self._disk_text: str = ""

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Static("no file open — pick one in the tree (or /open path)", id="editor-title")
        yield TextArea("", id="editor-body", show_line_numbers=True)

    def on_mount(self) -> None:
        try:
            self.body.theme = THEME
        except Exception:
            pass  # theme name differs across textual versions; default is fine
        self.body.read_only = True

    @property
    def body(self) -> TextArea:
        return self.query_one("#editor-body", TextArea)

    @property
    def title_widget(self) -> Static:
        return self.query_one("#editor-title", Static)

    # ------------------------------------------------------------------- state
    @property
    def is_dirty(self) -> bool:
        return self.current_path is not None and self.body.text != self._disk_text

    def _render_title(self) -> None:
        if self.current_path is None:
            self.title_widget.update("no file open — pick one in the tree (or /open path)")
            return
        try:
            rel = self.current_path.relative_to(self.root)
        except ValueError:
            rel = self.current_path
        mark = " ●" if self.is_dirty else ""
        self.title_widget.update(f"{rel}{mark}")

    # ----------------------------------------------------------------- actions
    def open_path(self, path: pathlib.Path | str) -> bool:
        target = pathlib.Path(path)
        if not target.is_absolute():
            target = self.root / target
        try:
            text = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            self.title_widget.update(f"cannot open {target.name}: {exc}")
            return False
        self.current_path = target.resolve()
        self._disk_text = text
        self.body.read_only = False
        self.body.load_text(text)
        self.body.language = language_for(self.current_path)
        self.body.cursor_location = (0, 0)
        self._render_title()
        return True

    def reload(self) -> None:
        if self.current_path and self.current_path.exists():
            self.open_path(self.current_path)

    def action_save(self) -> None:
        self.save()

    def save(self) -> bool:
        if self.current_path is None:
            self.notify("No file open", severity="warning")
            return False
        try:
            self.current_path.write_text(self.body.text, encoding="utf-8")
        except OSError as exc:
            self.notify(f"Save failed: {exc}", severity="error")
            return False
        self._disk_text = self.body.text
        self._render_title()
        self.notify(f"Saved {self.current_path.name}", timeout=2)
        return True

    # Called by the app when a tool touched a file on disk.
    def refresh_if_open(self, path: pathlib.Path) -> None:
        if self.current_path is None:
            return
        try:
            same = pathlib.Path(path).resolve() == self.current_path
        except OSError:
            return
        if not same:
            return
        if self.is_dirty:
            self.notify(
                "File changed on disk by the agent — your unsaved buffer was kept. Ctrl+S overwrites.",
                severity="warning",
                timeout=6,
            )
            return
        self.reload()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._render_title()
