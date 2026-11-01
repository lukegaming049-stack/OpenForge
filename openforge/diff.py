"""Unified diffs for approval previews and tool results."""

from __future__ import annotations

import difflib


def unified(old: str, new: str, path: str, context: int = 3) -> str:
    """A readable unified diff, or a placeholder when nothing changed."""
    if old == new:
        return f"(no changes to {path})"
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=context,
        lineterm="\n",
    )
    text = "".join(diff)
    # difflib can emit lines without trailing newlines; keep it tidy.
    return text if text.endswith("\n") else text + "\n"


def counts(old: str, new: str) -> tuple[int, int]:
    """(added, removed) line counts."""
    added = removed = 0
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(), n=0, lineterm=""):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def summarize(old: str, new: str) -> str:
    added, removed = counts(old, new)
    if added == removed == 0:
        return "no change"
    return f"+{added} -{removed}"
