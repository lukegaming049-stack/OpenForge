"""File tools: read, write, edit, list, glob, grep.

Every path goes through :meth:`ToolContext.resolve`, so a model cannot wander
out of the workspace by asking for ``../../etc/passwd``.
"""

from __future__ import annotations

import fnmatch
import os
import pathlib
import re
from typing import Any

from ..diff import summarize, unified
from .context import ApprovalRequest, ToolContext, ToolError

IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".idea",
    ".next",
    "target",
    "vendor",
    ".terraform",
}
MAX_FILE_BYTES = 4_000_000


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def read_text(path: pathlib.Path) -> str:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ToolError(f"{path.name} is larger than {MAX_FILE_BYTES // 1_000_000} MB")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ToolError(f"cannot read {path}: {exc}") from exc


# ------------------------------------------------------------------ read_file
def read_file(
    ctx: ToolContext, path: str, offset: int = 1, limit: int = 400, **_: Any
) -> str:
    target = ctx.resolve(path, must_exist=True)
    if target.is_dir():
        raise ToolError(f"{ctx.relative(target)} is a directory — use list_dir")
    text = read_text(target)
    lines = text.splitlines()
    start = max(_as_int(offset, 1), 1)
    count = max(_as_int(limit, 400), 1)
    window = lines[start - 1 : start - 1 + count]
    if not window and lines:
        raise ToolError(f"offset {start} is past the end of {ctx.relative(target)} ({len(lines)} lines)")
    body = "\n".join(f"{start + i:>6}| {line}" for i, line in enumerate(window))
    total = len(lines)
    if start > 1 or start - 1 + count < total:
        body += (
            f"\n... (lines {start}-{start + len(window) - 1} of {total}; "
            f"call read_file again with offset to see more)"
        )
    return body if body else f"({ctx.relative(target)} is empty)"


# ----------------------------------------------------------------- write_file
def write_file(ctx: ToolContext, path: str, content: str = "", **_: Any) -> str:
    target = ctx.resolve(path)
    rel = ctx.relative(target)
    if target.exists() and target.is_dir():
        raise ToolError(f"{rel} is a directory")
    old = read_text(target) if target.exists() else ""
    new = content if isinstance(content, str) else str(content)
    if old == new:
        return f"{rel} already has that exact content — nothing written."
    added, removed = _counts(old, new)
    request = ApprovalRequest(
        kind="write",
        title=f"write {rel}",
        detail=unified(old, new, rel),
        path=rel,
        added=added,
        removed=removed,
    )
    if not ctx.request(request):
        return f"User denied writing {rel}. Do not retry without changing your approach."
    if ctx.dry_run:
        return f"DRY RUN: would write {rel} ({summarize(old, new)})"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(new, encoding="utf-8")
    ctx.notify_change(target)
    verb = "updated" if old else "created"
    return f"{verb} {rel} ({summarize(old, new)})"


# ------------------------------------------------------------------ edit_file
def edit_file(
    ctx: ToolContext,
    path: str,
    old_string: str,
    new_string: str = "",
    replace_all: bool = False,
    **_: Any,
) -> str:
    target = ctx.resolve(path, must_exist=True)
    rel = ctx.relative(target)
    text = read_text(target)
    if old_string == "":
        raise ToolError("old_string is empty — use write_file to create or replace a whole file")
    occurrences = text.count(old_string)
    if occurrences == 0:
        hint = _closest_line(text, old_string)
        raise ToolError(
            f"old_string not found in {rel}.{hint}\n"
            "Read the file again and copy the exact text, including indentation."
        )
    if occurrences > 1 and not _as_bool(replace_all):
        lines = [i + 1 for i, line in enumerate(text.splitlines()) if old_string in line]
        raise ToolError(
            f"old_string appears {occurrences} times in {rel} "
            f"(lines {', '.join(map(str, lines[:10]))}). "
            "Include more surrounding context, or pass replace_all=true."
        )
    new_text = (
        text.replace(old_string, new_string)
        if _as_bool(replace_all)
        else text.replace(old_string, new_string, 1)
    )
    added, removed = _counts(text, new_text)
    request = ApprovalRequest(
        kind="write",
        title=f"edit {rel}",
        detail=unified(text, new_text, rel),
        path=rel,
        added=added,
        removed=removed,
    )
    if not ctx.request(request):
        return f"User denied editing {rel}. Do not retry without changing your approach."
    if ctx.dry_run:
        return f"DRY RUN: would edit {rel} ({summarize(text, new_text)})"
    target.write_text(new_text, encoding="utf-8")
    ctx.notify_change(target)
    times = f" ({occurrences} occurrences)" if _as_bool(replace_all) and occurrences > 1 else ""
    return f"edited {rel} ({summarize(text, new_text)}){times}"


def _counts(old: str, new: str) -> tuple[int, int]:
    from ..diff import counts

    return counts(old, new)


def _closest_line(text: str, needle: str) -> str:
    """Cheap fuzzy hint so the model can self-correct in one step."""
    probe = needle.strip().splitlines()[0].strip() if needle.strip() else ""
    if not probe:
        return ""
    lines = text.splitlines()
    # Try the whole probe, then progressively shorter prefixes, so a typo in the
    # tail still points at the right place.
    for width in (40, 24, 16, 10, 6):
        fragment = probe[:width]
        if len(fragment) < 4:
            break
        for i, line in enumerate(lines, 1):
            if fragment in line:
                return f"\nClosest match at line {i}: {line.strip()[:120]}"
    return ""


# ------------------------------------------------------------------- list_dir
def list_dir(ctx: ToolContext, path: str = ".", depth: int = 1, **_: Any) -> str:
    target = ctx.resolve(path, must_exist=True)
    if not target.is_dir():
        return read_file(ctx, path)
    max_depth = min(max(_as_int(depth, 1), 1), 4)
    root = target
    entries: list[str] = []
    truncated = False

    def walk(directory: pathlib.Path, level: int, prefix: str) -> None:
        nonlocal truncated
        try:
            children = sorted(
                directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
            )
        except PermissionError:
            entries.append(f"{prefix}[permission denied]")
            return
        for child in children:
            if len(entries) >= 400:
                truncated = True
                return
            if child.name in IGNORE_DIRS:
                entries.append(f"{prefix}{child.name}/  (ignored)")
                continue
            if child.is_dir():
                entries.append(f"{prefix}{child.name}/")
                if level < max_depth:
                    walk(child, level + 1, prefix + "  ")
            else:
                try:
                    size = child.stat().st_size
                except OSError:
                    size = 0
                entries.append(f"{prefix}{child.name}  ({_human(size)})")

    walk(root, 1, "")
    header = f"{ctx.relative(root) or '.'}/"
    body = "\n".join(entries) or "(empty)"
    if truncated:
        body += "\n... (truncated at 400 entries)"
    return f"{header}\n{body}"


def _human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


# --------------------------------------------------------------------- glob
def glob_files(ctx: ToolContext, pattern: str = "**/*", path: str = ".", **_: Any) -> str:
    base = ctx.resolve(path, must_exist=True)
    if not base.is_dir():
        base = base.parent
    matches: list[str] = []
    for candidate in base.glob(pattern):
        if any(part in IGNORE_DIRS for part in candidate.parts):
            continue
        if candidate.is_file():
            matches.append(ctx.relative(candidate))
        if len(matches) >= 300:
            break
    if not matches:
        return f"no files matching '{pattern}' under {ctx.relative(base) or '.'}"
    matches.sort()
    return "\n".join(matches[:300]) + (
        "\n... (truncated at 300 matches)" if len(matches) >= 300 else ""
    )


# --------------------------------------------------------------------- grep
def grep(
    ctx: ToolContext,
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    max_results: int = 60,
    ignore_case: bool = False,
    **_: Any,
) -> str:
    base = ctx.resolve(path, must_exist=True)
    try:
        regex = re.compile(pattern, re.IGNORECASE if _as_bool(ignore_case) else 0)
    except re.error as exc:
        raise ToolError(f"invalid regex '{pattern}': {exc}") from exc

    limit = max(_as_int(max_results, 60), 1)
    hits: list[str] = []
    files_scanned = 0

    def scan(file: pathlib.Path) -> None:
        nonlocal files_scanned
        if len(hits) >= limit:
            return
        try:
            if file.stat().st_size > 2_000_000:
                return
            data = file.read_bytes()
        except OSError:
            return
        if b"\x00" in data[:2048]:
            return
        files_scanned += 1
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            return
        for lineno, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                hits.append(f"{ctx.relative(file)}:{lineno}: {line.strip()[:200]}")
                if len(hits) >= limit:
                    return

    if base.is_file():
        scan(base)
    else:
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            for name in sorted(filenames):
                if glob and not fnmatch.fnmatch(name, glob):
                    continue
                scan(pathlib.Path(dirpath) / name)
                if len(hits) >= limit:
                    break

    if not hits:
        return f"no matches for /{pattern}/ ({files_scanned} files scanned)"
    body = "\n".join(hits)
    if len(hits) >= limit:
        body += f"\n... (stopped at {limit} matches)"
    return body
