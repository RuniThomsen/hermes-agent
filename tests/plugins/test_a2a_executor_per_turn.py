"""A2A turns run on a thread executor, not the gateway asyncio loop (#1406)."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.a2a import adapter as a2a_adapter
from plugins.platforms.a2a import protocol


def _adapter():
    adapter = a2a_adapter.A2AAdapter(PlatformConfig(enabled=True))
    adapter._loop = object()  # present so _prepare_task admits; must not be used
    adapter._message_handler = object()
    return adapter


def _pending(adapter, *, task_id: str, context_id: str, peer: str = "peer"):
    fut = adapter._add_pending(task_id, context_id)
    event = MessageEvent(
        text=f"turn-{task_id}",
        message_type=MessageType.TEXT,
        message_id=task_id,
        source=adapter.build_source(
            chat_id=context_id, chat_name=f"a2a:{peer}", chat_type="dm",
            user_id=peer, user_name=peer,
        ),
    )
    adapter.tasks.create(task_id, context_id, peer, *adapter._scope_for_agent(None))
    return {
        "task_id": task_id,
        "context_id": context_id,
        "peer": peer,
        "future": fut,
        "created_iso": "2026-09-11T00:00:00Z",
        "started": time.time(),
        "event": event,
    }


def test_start_pending_does_not_marshal_handle_message_onto_gateway_loop(monkeypatch):
    adapter = _adapter()
    marshaled = []

    def boom(*_a, **_k):
        marshaled.append(True)
        raise AssertionError("gateway loop must not receive handle_message")

    monkeypatch.setattr(a2a_adapter.asyncio, "run_coroutine_threadsafe", boom)

    called = {"n": 0}

    async def handle(_event):
        called["n"] += 1

    adapter.handle_message = handle  # type: ignore[method-assign]
    adapter._sync_conversation = lambda pending: pending["task_id"]
    pending = _pending(adapter, task_id="task-exec-1", context_id="ctx-exec-1")
    adapter._start_pending(pending)
    rec = adapter.tasks.get("task-exec-1")
    assert rec is not None
    assert rec["state"] == protocol.STATE_WORKING
    pending["future"].result(timeout=2)
    assert marshaled == []
    assert called["n"] == 0
    adapter._runner_executor().shutdown(wait=True)


def test_sibling_context_completes_while_other_turn_sleeps():
    adapter = _adapter()
    order: list[str] = []
    release_a = threading.Event()

    def turn(pending):
        ctx = pending["context_id"]
        if ctx == "ctx-slow":
            release_a.wait(timeout=2)
            order.append("slow")
            return "slow-done"
        order.append("fast")
        release_a.set()
        return "fast-done"

    adapter._sync_conversation = turn
    slow = _pending(adapter, task_id="task-slow", context_id="ctx-slow")
    fast = _pending(adapter, task_id="task-fast", context_id="ctx-fast")
    adapter._start_pending(slow)
    t0 = time.monotonic()
    adapter._start_pending(fast)
    assert fast["future"].result(timeout=2) == (protocol.STATE_COMPLETED, "fast-done")
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0
    assert "fast" in order
    assert slow["future"].result(timeout=2) == (protocol.STATE_COMPLETED, "slow-done")
    assert order[0] == "fast"
    adapter._runner_executor().shutdown(wait=True)


def test_same_context_turns_are_serial():
    adapter = _adapter()
    overlapping = {"n": 0}
    max_overlap = {"n": 0}
    gate = threading.Event()
    started = threading.Event()

    def turn(pending):
        overlapping["n"] += 1
        max_overlap["n"] = max(max_overlap["n"], overlapping["n"])
        if pending["task_id"] == "task-a":
            started.set()
            gate.wait(timeout=2)
        overlapping["n"] -= 1
        return pending["task_id"]

    adapter._sync_conversation = turn
    first = _pending(adapter, task_id="task-a", context_id="ctx-shared")
    second = _pending(adapter, task_id="task-b", context_id="ctx-shared")
    adapter._start_pending(first)
    assert started.wait(timeout=2)
    adapter._start_pending(second)
    time.sleep(0.05)
    assert max_overlap["n"] == 1
    gate.set()
    first["future"].result(timeout=2)
    second["future"].result(timeout=2)
    assert max_overlap["n"] == 1
    adapter._runner_executor().shutdown(wait=True)
