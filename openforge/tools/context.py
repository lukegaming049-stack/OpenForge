"""Execution context shared by every tool: workspace root, approvals, guards."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Callable


class ToolError(RuntimeError):
    """A tool failure the model should read and react to (not a crash)."""


@dataclass
class ApprovalRequest:
    """What the user is being asked to allow."""

    kind: str  # "write" | "shell"
    title: str
    detail: str = ""
    path: str | None = None
    added: int = 0
    removed: int = 0

    def render(self) -> str:
        head = f"{self.title}"
        if self.kind == "write" and (self.added or self.removed):
            head += f"  (+{self.added} -{self.removed})"
        return f"{head}\n{self.detail}".rstrip()


ApprovalFn = Callable[[ApprovalRequest], bool]
ChangeFn = Callable[[pathlib.Path], None]


def _auto_deny(request: ApprovalRequest) -> bool:  # pragma: no cover - default
    return False


@dataclass
class ToolContext:
    """Everything a tool needs to know about the run."""

    root: pathlib.Path
    approve: ApprovalFn = _auto_deny
    allow_outside_root: bool = False
    dry_run: bool = False
    on_change: ChangeFn | None = None
    approved_writes: list[str] = field(default_factory=list)
    denied: list[str] = field(default_factory=list)

    # ------------------------------------------------------------ path guard
    def resolve(self, path: str | None, *, must_exist: bool = False) -> pathlib.Path:
        """Resolve ``path`` against the workspace root, refusing escapes."""
        raw = (path or ".").strip() or "."
        candidate = pathlib.Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        root = self.root.resolve()
        inside = resolved == root or root in resolved.parents
        if not inside and not self.allow_outside_root:
            raise ToolError(
                f"refusing to touch '{raw}': it is outside the workspace ({root}). "
                "Set allow_outside_root = true in config.toml if you really mean it."
            )
        if must_exist and not resolved.exists():
            raise ToolError(f"no such file or directory: {self.relative(resolved)}")
        return resolved

    def relative(self, path: pathlib.Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(path)

    # ------------------------------------------------------------- approvals
    def request(self, request: ApprovalRequest) -> bool:
        """Ask the user (or the configured policy) to allow an action."""
        allowed = bool(self.approve(request))
        if allowed:
            self.approved_writes.append(request.title)
        else:
            self.denied.append(request.title)
        return allowed

    def notify_change(self, path: pathlib.Path) -> None:
        if self.on_change:
            try:
                self.on_change(path)
            except Exception:  # pragma: no cover - UI callback must never break a run
                pass
