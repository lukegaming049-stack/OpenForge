"""The agent loop: native tool calls, the text protocol, guards, approvals."""

import json
import pathlib

from openforge.agent import Agent, parse_tool_blocks
from openforge.config import Config
from openforge.providers.base import ToolCall, Turn
from openforge.tools import ToolContext


def make_agent(provider, root, approve=lambda r: True, **kw) -> Agent:
    ctx = ToolContext(root=pathlib.Path(root), approve=approve)
    return Agent(provider, "fake-model", ctx, config=Config(temperature=0.0), **kw)


# --------------------------------------------------------------- text protocol
def test_parse_tool_blocks_single():
    prose, calls = parse_tool_blocks(
        'Reading it now.\n```tool\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```'
    )
    assert prose == "Reading it now."
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "a.py"}


def test_parse_tool_blocks_multiple_and_json_blocks():
    text = (
        '```tool\n{"name": "read_file", "arguments": {"path": "a"}}\n```\n'
        '```json\n{"name": "list_dir", "arguments": {"path": "."}}\n```\n'
        "then I will stop"
    )
    prose, calls = parse_tool_blocks(text)
    assert [c.name for c in calls] == ["read_file", "list_dir"]
    assert "then I will stop" in prose
    assert "```" not in prose


def test_parse_tool_blocks_ignores_garbage():
    prose, calls = parse_tool_blocks("here is code:\n```tool\nnot json at all\n```\nbye")
    assert calls == []
    assert "bye" in prose


def test_parse_tool_blocks_accepts_list_payload():
    _prose, calls = parse_tool_blocks(
        '```tool\n[{"name": "list_dir", "arguments": {}}, {"name": "glob", "arguments": {"pattern": "*"}}]\n```'
    )
    assert [c.name for c in calls] == ["list_dir", "glob"]


def test_text_mode_loop_writes_then_finishes(scripted, workspace):
    provider = scripted(
        [
            Turn(text='```tool\n{"name": "write_file", "arguments": {"path": "hello.txt", "content": "hi\\n"}}\n```'),
            Turn(text="Created hello.txt."),
        ],
        supports_native_tools=False,
    )
    events = []
    agent = make_agent(provider, workspace, tool_mode="auto", on_event=events.append)
    assert agent.tool_mode == "text"
    answer = agent.run("create hello.txt")

    assert answer == "Created hello.txt."
    assert (workspace / "hello.txt").read_text() == "hi\n"
    # tool schemas are NOT sent in text mode, the catalogue is in the system prompt
    assert provider.requests[0]["tools"] is None
    system = provider.requests[0]["messages"][0]
    assert system["role"] == "system" and "read_file" in system["content"]
    # the second request must carry the tool result back to the model
    joined = json.dumps(provider.requests[1]["messages"])
    assert "Tool results" in joined and "created hello.txt" in joined
    kinds = [e.kind for e in events]
    assert "tool_start" in kinds and "tool_result" in kinds and "done" in kinds


# ------------------------------------------------------------- native tool use
def test_native_mode_loop(scripted, workspace):
    provider = scripted(
        [
            Turn(
                text="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="write_file",
                        arguments={"path": "native.txt", "content": "ok"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            Turn(text="done"),
        ]
    )
    agent = make_agent(provider, workspace)
    assert agent.tool_mode == "native"
    assert agent.run("write native.txt") == "done"
    assert (workspace / "native.txt").read_text() == "ok"

    first = provider.requests[0]
    assert first["tools"] and first["tools"][0]["name"] == "read_file"
    # assistant turn with tool_calls, then a role=tool message with the id
    tail = provider.requests[1]["messages"][-2:]
    assert tail[0]["role"] == "assistant" and tail[0]["tool_calls"][0]["id"] == "call_1"
    assert tail[1]["role"] == "tool" and tail[1]["tool_call_id"] == "call_1"


def test_provider_without_native_tools_falls_back_to_text(scripted, workspace):
    provider = scripted([Turn(text="nothing to do")], supports_native_tools=False)
    agent = make_agent(provider, workspace, tool_mode="native")
    assert agent.tool_mode == "text"
    assert agent.run("hi") == "nothing to do"


# ------------------------------------------------------------------- guardrails
def test_denied_write_is_reported_to_the_model(scripted, workspace):
    provider = scripted(
        [
            Turn(
                text="",
                tool_calls=[
                    ToolCall(id="c1", name="write_file", arguments={"path": "x.txt", "content": "1"})
                ],
            ),
            Turn(text="Understood, I won't."),
        ]
    )
    agent = make_agent(provider, workspace, approve=lambda r: False)
    agent.run("write x.txt")
    assert not (workspace / "x.txt").exists()
    tool_msg = provider.requests[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and "denied" in tool_msg["content"].lower()


def test_identical_repeated_calls_are_stopped(scripted, workspace):
    call = ToolCall(id="c", name="list_dir", arguments={"path": "."})
    provider = scripted([Turn(tool_calls=[call]) for _ in range(10)])
    agent = make_agent(provider, workspace)
    answer = agent.run("loop forever")
    assert "identical arguments" in answer
    assert len(provider.requests) <= 4


def test_step_cap(scripted, workspace):
    turns = [
        Turn(tool_calls=[ToolCall(id=f"c{i}", name="list_dir", arguments={"path": f"./{i}"})])
        for i in range(10)
    ]
    for i in range(10):
        (workspace / str(i)).mkdir()
    agent = make_agent(scripted(turns), workspace, max_steps=3)
    answer = agent.run("go")
    assert "Stopped after 3 steps" in answer


def test_mention_attaches_file(scripted, workspace):
    provider = scripted([Turn(text="seen")])
    agent = make_agent(provider, workspace)
    agent.run("what does @src/app.py do?")
    user_msg = provider.requests[0]["messages"][-1]["content"]
    assert "def hello" in user_msg
    assert "Files I attached" in user_msg


def test_project_notes_are_loaded(scripted, workspace):
    (workspace / "AGENTS.md").write_text("Always use tabs.", encoding="utf-8")
    provider = scripted([Turn(text="ok")])
    agent = make_agent(provider, workspace)
    assert "Always use tabs." in agent.system_prompt


def test_usage_is_accumulated(scripted, workspace):
    provider = scripted([Turn(text="done", usage={"prompt_tokens": 11, "completion_tokens": 7})])
    agent = make_agent(provider, workspace)
    agent.run("hi")
    assert agent.usage == {"prompt_tokens": 11, "completion_tokens": 7}


def test_history_is_reusable(scripted, workspace):
    provider = scripted([Turn(text="first"), Turn(text="second")])
    agent = make_agent(provider, workspace)
    agent.run("one")
    agent.run("two")
    roles = [m["role"] for m in provider.requests[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert agent.export()["messages"][0]["content"] == "one"
