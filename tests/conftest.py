import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from openforge.providers.base import Provider, Turn  # noqa: E402


class ScriptedProvider(Provider):
    """Hands back canned turns and records what it was asked."""

    supports_native_tools = True

    def __init__(self, turns, supports_native_tools: bool = True):
        super().__init__("scripted", "fake-model")
        self.turns = list(turns)
        self.supports_native_tools = supports_native_tools
        self.requests: list[dict] = []

    def chat(self, messages, tools=None, *, on_delta=None, temperature=None, max_tokens=None):
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools})
        if not self.turns:
            return Turn(text="(out of scripted turns)")
        turn = self.turns.pop(0)
        if on_delta and turn.text:
            on_delta(turn.text)
        return turn


@pytest.fixture
def scripted():
    return ScriptedProvider


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
    return tmp_path
