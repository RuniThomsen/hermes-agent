"""A2A out-of-process cron delivery uses standalone_sender_fn, not live adapter.send."""

from __future__ import annotations

from types import SimpleNamespace

from plugins.platforms.a2a import tools
from plugins.platforms.a2a.tools import (
    _loopback_direct_url,
    _standalone_send_sync,
)


def test_loopback_direct_url_adds_direct_flag():
    assert _loopback_direct_url("http://127.0.0.1:3979/") == "http://127.0.0.1:3979/?direct=1"
    assert _loopback_direct_url("http://127.0.0.1:3979") == "http://127.0.0.1:3979/?direct=1"
    already = "http://127.0.0.1:3979/?direct=1"
    assert _loopback_direct_url(already) == already
    public = "https://example.azure-api.net/peer/"
    assert _loopback_direct_url(public) == "https://example.azure-api.net/peer/?direct=1"


def test_standalone_send_refuses_anonymous(monkeypatch):
    monkeypatch.setattr(tools, "_configured_peers", lambda: {})
    pconfig = SimpleNamespace(extra={"local_peer_url": "http://127.0.0.1:3979/"})
    result = _standalone_send_sync(pconfig, "ctx-origin", "morning line")
    assert "error" in result
    assert "Bearer" in result["error"]


def test_standalone_send_posts_to_direct_url_with_bearer(monkeypatch, tmp_path):
    token_file = tmp_path / "peer.token"
    token_file.write_text("loopback-token\n", encoding="utf-8")
    captured = {}

    def fake_post(url, body, headers, timeout):
        captured["url"] = url
        captured["body"] = body
        captured["headers"] = headers
        captured["timeout"] = timeout
        return {"jsonrpc": "2.0", "id": body["id"], "result": {"id": "task-recv-1", "contextId": "ctx-origin", "status": {"state": "submitted"}}}

    monkeypatch.setattr(tools, "_configured_peers", lambda: {
        "peer": {
            "url": "http://127.0.0.1:3979/",
            "timeout": 15,
            "auth": {"type": "bearer", "token_file": str(token_file)},
        }
    })
    monkeypatch.setattr(tools, "_http_post_json", fake_post)
    pconfig = SimpleNamespace(extra={"local_peer_url": "http://127.0.0.1:3979/", "local_first_send": True})
    result = _standalone_send_sync(pconfig, "ctx-origin", "Cronjob Response: Morning house")
    assert result.get("success") is True
    assert result.get("message_id") == "task-recv-1"
    assert captured["url"] == "http://127.0.0.1:3979/?direct=1"
    assert captured["headers"].get("Authorization") == "Bearer loopback-token"
    assert captured["body"]["method"] == "SendMessage"
    assert captured["body"]["params"]["configuration"]["blocking"] is False
    msg = captured["body"]["params"]["message"]
    assert msg["contextId"] == "ctx-origin"


def test_register_wires_standalone_sender():
    captured = {}

    class Ctx:
        def register_tool(self, **kwargs):
            return None

        def register_platform(self, **kwargs):
            captured.update(kwargs)

    from plugins.platforms.a2a import register

    register(Ctx())
    assert captured.get("name") == "a2a"
    assert captured.get("standalone_sender_fn") is tools._standalone_send
