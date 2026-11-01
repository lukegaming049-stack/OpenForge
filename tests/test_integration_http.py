"""End-to-end: real HTTP (SSE) -> provider -> agent loop -> file on disk.

A tiny loopback server stands in for the vendor, so this exercises the wire
format, streaming parser, tool dispatch and approval path together.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from openforge.agent import Agent
from openforge.config import Config, ProviderConfig
from openforge.providers import build_provider
from openforge.tools import ToolContext


class FakeVendor(BaseHTTPRequestHandler):
    script: list[dict] = []
    requests: list[dict] = []

    def log_message(self, *args):  # silence test output
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        FakeVendor.requests.append(payload)
        index = len(FakeVendor.requests) - 1
        step = FakeVendor.script[min(index, len(FakeVendor.script) - 1)]

        if not payload.get("stream"):
            body = json.dumps(
                {
                    "choices": [
                        {"message": {"role": "assistant", "content": step["text"]}, "finish_reason": "stop"}
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for chunk in step["chunks"]:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@pytest.fixture
def vendor():
    FakeVendor.requests = []
    FakeVendor.script = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeVendor)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _delta(content=None, tool_calls=None, finish=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def test_streaming_tool_call_round_trip(vendor, workspace):
    FakeVendor.script = [
        {
            "text": "",
            "chunks": [
                _delta(content="Writing it now. "),
                _delta(tool_calls=[{"index": 0, "id": "c1", "function": {"name": "write_file", "arguments": '{"path": "gen.py", '}}]),
                _delta(tool_calls=[{"index": 0, "function": {"arguments": '"content": "print(1)\\n"}'}}], finish="tool_calls"),
            ],
        },
        {"text": "Done — gen.py prints 1.", "chunks": [_delta(content="Done — "), _delta(content="gen.py prints 1.", finish="stop")]},
    ]

    pc = ProviderConfig(
        name="fake", type="openai_compat", base_url=f"http://127.0.0.1:{vendor}/v1", api_key="sk-test"
    )
    provider = build_provider(pc, "fake-model")
    ctx = ToolContext(root=workspace, approve=lambda request: True)
    events = []
    agent = Agent(provider, "fake-model", ctx, config=Config(temperature=0.0), on_event=events.append)

    answer = agent.run("create gen.py")

    assert answer.startswith("Done")
    assert (workspace / "gen.py").read_text() == "print(1)\n"
    # the tool result went back on the wire as a role=tool message
    second = FakeVendor.requests[1]["messages"]
    assert second[-1]["role"] == "tool" and "created gen.py" in second[-1]["content"]
    # and the tool schema was advertised on the first call
    assert FakeVendor.requests[0]["tools"][0]["function"]["name"] == "read_file"
    assert any(e.kind == "delta" for e in events)
    provider.close()


def test_http_error_becomes_a_readable_message(vendor, workspace):
    from openforge.providers.base import ProviderError

    pc = ProviderConfig(
        name="fake", type="openai_compat", base_url=f"http://127.0.0.1:{vendor}/v1", api_key="sk-test"
    )
    provider = build_provider(pc, "fake-model")

    class Boom(FakeVendor):
        def do_POST(self):
            body = b'{"error":{"message":"bad model"}}'
            self.send_response(404)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Boom)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider.base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        with pytest.raises(ProviderError) as exc:
            provider.chat([{"role": "user", "content": "hi"}])
        assert "404" in str(exc.value) and "not found" in str(exc.value)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
