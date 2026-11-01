"""Session persistence — history survives a restart, stored locally as JSON."""

from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Any


def state_dir() -> pathlib.Path:
    base = os.environ.get("XDG_STATE_HOME")
    root = pathlib.Path(base).expanduser() if base else pathlib.Path.home() / ".local" / "state"
    return root / "openforge"


def _path_for(session: str | None) -> pathlib.Path:
    if session:
        p = pathlib.Path(session).expanduser()
        return p if p.suffix == ".json" else p.with_suffix(".json")
    return state_dir() / "last_session.json"


def save(payload: dict[str, Any], session: str | None = None) -> pathlib.Path:
    path = _path_for(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def load(session: str | None = None) -> dict[str, Any] | None:
    path = _path_for(session)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def last_session_path() -> pathlib.Path:
    return _path_for(None)
