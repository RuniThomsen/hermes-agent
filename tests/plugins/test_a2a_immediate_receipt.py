"""Immediate A2A SendMessage receipt — the RPC must not wait for the agent turn."""

from __future__ import annotations

import asyncio
import threading
import time

from plugins.platforms.a2a import adapter as a2a_adapter
from plugins.platforms.a2a import protocol


def test_message_send_exposes_submitted_before_session_dispatch(monkeypatch):
    """blocking=false receipt is the minted Task; session work starts after HTTP write."""
    from gateway.config import PlatformConfig

    monkeypatch.delenv("A2A_REPLY_TIMEOUT", raising=False)
    adapter = a2a_adapter.A2AAdapter(PlatformConfig(enabled=True))
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    adapter._loop = loop
    adapter._message_handler = object()
    dispatched = {"n": 0}

    async def handle(_event):
        dispatched["n"] += 1

    adapter.handle_message = handle  # type: ignore[method-assign]
    params = {
        "message": protocol.text_message(protocol.ROLE_USER, "receipt-before-session"),
    }
    try:
        started = time.monotonic()
        response = adapter._rpc_message_send("send-admit", params, "peer")
        assert time.monotonic() - started < 1.0, "SendMessage held the RPC instead of returning a receipt"
        task = response["result"]
        assert task["status"]["state"] == protocol.STATE_SUBMITTED
        rec = adapter.tasks.get(task["id"])
        assert rec is not None
        assert rec["state"] == protocol.STATE_SUBMITTED
        assert dispatched["n"] == 0
        adapter._start_deferred_pending()
        deadline = time.monotonic() + 1
        rec = adapter.tasks.get(task["id"])
        while time.monotonic() < deadline:
            rec = adapter.tasks.get(task["id"])
            if rec and rec["state"] == protocol.STATE_WORKING:
                break
            time.sleep(0.01)
        assert rec is not None
        assert rec["state"] == protocol.STATE_WORKING
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(1)
        loop.close()
