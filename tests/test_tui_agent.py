"""TUI plumbing test: worker thread -> modal approval -> file on disk.

Skipped automatically when textual is not installed.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

textual = pytest.importorskip("textual")

from openforge.config import Config  # noqa: E402
from openforge.tui.app import ConfirmScreen, ForgeApp  # noqa: E402


class Vendor(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        Vendor.requests.append(payload)
        first = len(Vendor.requests) == 1
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        if first:
            chunks = [
                {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "write_file", "arguments": json.dumps({"path": "out.txt", "content": "hello\n"})}}]}, "finish_reason": "tool_calls"}]}
            ]
        else:
            chunks = [
                {"choices": [{"index": 0, "delta": {"content": "Wrote out.txt."}, "finish_reason": "stop"}]}
            ]
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def test_tui_agent_run_with_modal_approval(tmp_path):
    Vendor.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Vendor)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        f"""
default_model = "fake/test-model"
[providers.fake]
type = "openai_compat"
base_url = "http://127.0.0.1:{port}/v1"
api_key = "sk-test"
""",
        encoding="utf-8",
    )
    workspace = tmp_path / "repo"
    workspace.mkdir()
    cfg = Config.load(cfg_file)

    async def scenario() -> None:
        app = ForgeApp(cfg, root=workspace)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.prompt_input.value = "create out.txt"
            await pilot.press("enter")
            for _ in range(200):
                await pilot.pause()
                if isinstance(app.screen, ConfirmScreen):
                    await pilot.press("y")
                if not app._busy and (workspace / "out.txt").exists():
                    break
            await pilot.pause()
        assert (workspace / "out.txt").read_text() == "hello\n"
        assert len(Vendor.requests) == 2

    try:
        asyncio.run(scenario())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
