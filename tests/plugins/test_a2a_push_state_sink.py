"""#1406: SendMessage receipt, HTTP worker cap, and push of the Task on every state change.

Live adapter on a free loopback port, a local HTTP sink as the push receiver, no tasks/get.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from plugins.platforms.a2a import protocol


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _post(url: str, body: dict, timeout: float = 10.0):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def _send_body(text: str, configuration: dict) -> dict:
    return {"jsonrpc": "2.0", "id": "1", "method": "SendMessage",
            "params": {"message": protocol.text_message(protocol.ROLE_USER, text), "configuration": configuration}}


class _Sink:
    """Push receiver: records every POSTed JSON body."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []
        self.cond = threading.Condition()
        sink = self

        class _H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # noqa: A002
                pass

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode())
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
                with sink.cond:
                    sink.bodies.append(body)
                    sink.cond.notify_all()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/push/hermes"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def states(self) -> list[str]:
        with self.cond:
            return [b["task"]["status"]["state"] for b in self.bodies]

    def wait_for(self, state: str, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        with self.cond:
            while not any(b["task"]["status"]["state"] == state for b in self.bodies):
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self.cond.wait(left)
            return True

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _live_adapter(monkeypatch, release: threading.Event):
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a.adapter import A2AAdapter

    for var in ("A2A_BEARER_TOKEN", "A2A_PEER_TOKENS", "A2A_PUSH_SECRET"):
        monkeypatch.delenv(var, raising=False)  # localhost-only: loopback callbacks are allowed
    port = _free_port()
    monkeypatch.setenv("A2A_PORT", str(port))
    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={"port": port}))

    async def handle_message(event):
        await asyncio.to_thread(release.wait, 10)  # the agent turn, held until the test releases it
        await adapter.send(event.source.chat_id, "PUSHED REPLY: " + event.text, metadata={"notify": True})

    adapter.handle_message = handle_message  # type: ignore[method-assign]
    adapter._message_handler = object()
    gets = {"n": 0}
    real_get = adapter._rpc_tasks_get

    def counting_get(*a, **kw):
        gets["n"] += 1
        return real_get(*a, **kw)

    adapter._rpc_tasks_get = counting_get  # type: ignore[method-assign]
    return adapter, f"http://127.0.0.1:{port}/", gets


def _run_push_case(monkeypatch, configuration_for):
    sink, release = _Sink(), threading.Event()
    adapter, base, gets = _live_adapter(monkeypatch, release)
    out: dict = {}

    async def run():
        assert await adapter.connect() is True
        started = time.monotonic()
        status, resp = await asyncio.to_thread(_post, base, _send_body("hello push", configuration_for(sink.url)))
        out["receipt_s"] = time.monotonic() - started
        assert status == 200
        task = resp["result"]["task"]
        out["task"] = task
        # Receipt is on the wire while the turn is still held.
        assert task["status"]["state"] == protocol.STATE_SUBMITTED
        assert await asyncio.to_thread(sink.wait_for, protocol.STATE_WORKING)
        # Config persists while the task runs (A2A 1.0 §3.1.7): peeked, not popped.
        out["url_while_running"] = adapter.tasks.peek_push_url(task["id"])
        release.set()
        assert await asyncio.to_thread(sink.wait_for, protocol.STATE_COMPLETED)
        out["url_after_terminal"] = adapter.tasks.peek_push_url(task["id"])
        await adapter.disconnect()

    try:
        asyncio.run(run())
    finally:
        release.set()
        sink.close()
    return sink, gets, out


def test_push_v1_field(monkeypatch):
    sink, gets, out = _run_push_case(
        monkeypatch, lambda url: {"taskPushNotificationConfig": {"url": url}})
    assert out["receipt_s"] < 1.0
    assert sink.states() == [protocol.STATE_SUBMITTED, protocol.STATE_WORKING, protocol.STATE_COMPLETED]
    assert all(set(b) == {"task"} for b in sink.bodies)  # StreamResponse task member
    assert all(b["task"]["id"] == out["task"]["id"] for b in sink.bodies)
    completed = sink.bodies[-1]["task"]
    assert "PUSHED REPLY: " in protocol.extract_text(completed["status"]["message"])
    assert "PUSHED REPLY: " in protocol.extract_text(completed["artifacts"][0])
    assert out["url_while_running"] == sink.url
    assert out["url_after_terminal"] == ""
    assert gets["n"] == 0  # nobody polled


def test_push_compat_field(monkeypatch):
    """Seed's live wire: the JSON-RPC compat sibling pushNotificationConfig under configuration."""
    sink, gets, out = _run_push_case(
        monkeypatch, lambda url: {"pushNotificationConfig": {"url": url}})
    assert sink.states() == [protocol.STATE_SUBMITTED, protocol.STATE_WORKING, protocol.STATE_COMPLETED]
    assert "PUSHED REPLY: " in protocol.extract_text(sink.bodies[-1]["task"]["status"]["message"])
    assert out["url_after_terminal"] == ""
    assert gets["n"] == 0


def test_stale_state_dropped():
    """A WORKING that loses the race to COMPLETED is dropped, not delivered after it."""
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a.adapter import A2AAdapter

    adapter = A2AAdapter(PlatformConfig(enabled=True))
    sent: list[str] = []
    adapter._send_push_notification = (  # type: ignore[method-assign]
        lambda _tid, _url, payload: sent.append(payload["task"]["status"]["state"]))
    rec = adapter.tasks.create("task-race", "ctx-race", "peer")
    adapter.tasks.set_push_config("task-race", "http://127.0.0.1:9/push")
    adapter.tasks.on_change = None  # drive the two notifications by hand, in the losing order
    adapter.tasks.set_state("task-race", protocol.STATE_WORKING)
    working = adapter.tasks.get("task-race")
    done = adapter.tasks.complete("task-race", protocol.STATE_COMPLETED, "ok")
    adapter._push_task(done)
    adapter._push_task(working)
    deadline = time.monotonic() + 2
    while not sent and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)
    assert sent == [protocol.STATE_COMPLETED]
    assert (rec["seq"], working["seq"], done["seq"]) == (0, 1, 2)


def test_worker_cap_busy(monkeypatch):
    release = threading.Event()
    monkeypatch.setenv("A2A_MAX_HTTP_WORKERS", "2")
    adapter, base, _gets = _live_adapter(monkeypatch, release)
    port = int(base.rsplit(":", 1)[1].strip("/"))

    async def run():
        assert await adapter.connect() is True
        holders = [socket.create_connection(("127.0.0.1", port)) for _ in range(2)]  # idle: each holds a worker
        try:
            await asyncio.sleep(0.2)
            before = protocol.metrics.snapshot()["http_busy_rejects"]
            results = await asyncio.gather(*[asyncio.to_thread(
                _post, base, {"jsonrpc": "2.0", "id": str(i), "method": "GetTask", "params": {"id": "x"}}, 5)
                for i in range(10)])
            assert all(code == 503 and body["error"]["code"] == protocol.ERR_SERVER_BUSY for code, body in results)
            assert protocol.metrics.snapshot()["http_busy_rejects"] - before == 10
            assert adapter._httpd._worker_slots._value == 0  # still exactly the 2 held workers
        finally:
            for s in holders:
                s.close()
        await asyncio.sleep(0.3)
        code, body = await asyncio.to_thread(
            _post, base, {"jsonrpc": "2.0", "id": "after", "method": "GetTask", "params": {"id": "x"}})
        assert code == 200 and body["error"]["code"] == protocol.ERR_TASK_NOT_FOUND  # slots came back
        await adapter.disconnect()

    try:
        asyncio.run(run())
    finally:
        release.set()
