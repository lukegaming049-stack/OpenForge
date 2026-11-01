"""Shell tool.  Always approval-gated: nothing runs without a yes."""

from __future__ import annotations

import subprocess
from typing import Any

from .context import ApprovalRequest, ToolContext, ToolError

MAX_STREAM_CHARS = 12_000
HARD_TIMEOUT = 300


def run_bash(
    ctx: ToolContext, command: str, timeout: int = 120, cwd: str | None = None, **_: Any
) -> str:
    if not command or not command.strip():
        raise ToolError("command is empty")
    workdir = ctx.resolve(cwd) if cwd else ctx.root
    if not workdir.exists():
        raise ToolError(f"cwd does not exist: {ctx.relative(workdir)}")

    request = ApprovalRequest(
        kind="shell",
        title="run shell command",
        detail=f"$ {command}\n\n(cwd: {ctx.relative(workdir) or '.'})",
    )
    if not ctx.request(request):
        return "User denied running that command. Do not retry it; ask how they want to proceed."
    if ctx.dry_run:
        return f"DRY RUN: would run `{command}`"

    try:
        seconds = min(int(timeout), HARD_TIMEOUT)
    except (TypeError, ValueError):
        seconds = 120
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=seconds,
        )
    except subprocess.TimeoutExpired as exc:
        partial = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        return _format(None, partial, f"timed out after {seconds}s", truncated=partial)
    except OSError as exc:
        raise ToolError(f"cannot run command: {exc}") from exc

    return _format(proc.returncode, proc.stdout, proc.stderr)


def _format(code: int | None, stdout: str, stderr: str, truncated: str = "") -> str:
    parts: list[str] = []
    if code is not None:
        parts.append(f"exit code: {code}")
    if stdout.strip():
        parts.append("--- stdout ---\n" + _clip(stdout))
    if stderr.strip():
        parts.append("--- stderr ---\n" + _clip(stderr))
    if not stdout.strip() and not stderr.strip() and not truncated:
        parts.append("(no output)")
    return "\n".join(parts)


def _clip(text: str) -> str:
    if len(text) <= MAX_STREAM_CHARS:
        return text.rstrip()
    head = text[: MAX_STREAM_CHARS // 2]
    tail = text[-MAX_STREAM_CHARS // 2 :]
    return f"{head}\n... (truncated {len(text) - MAX_STREAM_CHARS} chars) ...\n{tail}".rstrip()
