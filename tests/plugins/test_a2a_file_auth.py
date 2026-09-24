"""Regression coverage for native A2A file-backed credentials."""
import os

import pytest

from plugins.platforms.a2a import tools


def test_file_backed_bearer(tmp_path):
    token = tmp_path / "bearer"
    token.write_text("x" * 40 + "\n")
    token.chmod(0o600)
    assert tools._auth_header({"type": "bearer", "token_file": str(token)}) == {
        "Authorization": "Bearer " + "x" * 40
    }


@pytest.mark.parametrize("content", ["", "short", "x" * 32 + "\ny", "x" * 4097])
def test_invalid_file_fails_closed(tmp_path, content):
    token = tmp_path / "bearer"
    token.write_text(content)
    token.chmod(0o600)
    with pytest.raises(ValueError):
        tools._auth_header({"type": "bearer", "token_file": str(token)})


def test_missing_file_fails_closed(tmp_path):
    with pytest.raises(ValueError):
        tools._auth_header({"type": "bearer", "token_file": str(tmp_path / "absent")})


@pytest.mark.skipif(os.name == "nt", reason="POSIX file permissions")
def test_public_file_fails_closed(tmp_path):
    token = tmp_path / "bearer"
    token.write_text("x" * 40)
    token.chmod(0o644)
    with pytest.raises(ValueError):
        tools._auth_header({"type": "bearer", "token_file": str(token)})


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
def test_symlink_fails_closed(tmp_path):
    token = tmp_path / "bearer"
    token.write_text("x" * 40)
    token.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(token)
    with pytest.raises(ValueError):
        tools._auth_header({"type": "bearer", "token_file": str(link)})


def test_inline_compatibility():
    assert tools._auth_header({"type": "bearer", "token": "legacy"}) == {"Authorization": "Bearer legacy"}


def test_query_discovery_url(monkeypatch):
    urls = []
    monkeypatch.setattr(tools, "_http_get_json", lambda url, *_: urls.append(url) or {})
    tools._fetch_card("http://127.0.0.1:3979/?direct=1", {}, 10)
    assert urls == ["http://127.0.0.1:3979/.well-known/agent-card.json?direct=1"]


def test_explicit_query_rpc_route_stays_direct():
    url = "http://127.0.0.1:3979/?direct=1"
    card = {"supportedInterfaces": [{"protocolBinding": "JSONRPC", "url": "https://public.invalid/receiver/"}]}
    assert tools._rpc_url(url, card) == url
