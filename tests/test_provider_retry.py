from __future__ import annotations

import requests

from butler.agent.provider import ModelProviderError, OllamaProvider


class _Resp:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload

    def close(self):
        return None


def test_provider_retries_then_succeeds(monkeypatch) -> None:
    calls = {"n": 0}

    def fake_post(self, url, json=None, timeout=None, stream=False):  # noqa: A002
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ReadTimeout("timeout")
        return _Resp(200, {"message": {"content": "ok"}})

    monkeypatch.setattr("butler.agent.provider.requests.Session.post", fake_post)

    p = OllamaProvider(base_url="http://127.0.0.1:11434", model="gemma:2b", timeout_seconds=1, retry_count=1, total_timeout_seconds=10)
    out = p.chat([{"role": "user", "content": "hi"}], temperature=0.0)
    assert out == "ok"
    assert calls["n"] == 2


def test_provider_budget_exhaustion(monkeypatch) -> None:
    def fake_post(self, url, json=None, timeout=None, stream=False):  # noqa: A002
        raise requests.exceptions.ReadTimeout("timeout")

    # Force time to jump beyond budget after first attempt.
    t = {"v": 0.0}

    def fake_time():
        t["v"] += 500.0
        return t["v"]

    monkeypatch.setattr("butler.agent.provider.requests.Session.post", fake_post)
    monkeypatch.setattr("butler.agent.provider.time.time", fake_time)
    monkeypatch.setattr("butler.agent.provider.time.sleep", lambda s: None)

    p = OllamaProvider(base_url="http://127.0.0.1:11434", model="gemma:2b", timeout_seconds=1, retry_count=5, total_timeout_seconds=220)
    try:
        p.chat([{"role": "user", "content": "hi"}], temperature=0.0)
        assert False, "expected ModelProviderError"
    except ModelProviderError as e:
        assert "budget" in str(e).lower()
