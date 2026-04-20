from __future__ import annotations

from dataclasses import dataclass

from butler.config import ButlerConfig
from butler.db import ButlerDB
from butler.sandbox import PathSandbox
from butler.tools.base import ToolContext
from butler.tools.impl import web


@dataclass
class _FakeDDGS:
    results: list[dict]
    calls: int = 0

    def text(self, query: str, max_results: int = 15):
        self.calls += 1
        return list(self.results)


def _ctx(tmp_path, monkeypatch) -> ToolContext:
    monkeypatch.setenv("BUTLER_HOME", str(tmp_path))
    cfg = ButlerConfig()
    db = ButlerDB.open(cfg)
    return ToolContext(config=cfg, db=db, sandbox=PathSandbox.from_strings([]))


def test_web_search_cached_hit_returns_without_sleep(tmp_path, monkeypatch) -> None:
    ctx = _ctx(tmp_path, monkeypatch)
    web._web_cache.clear()
    web._query_last_call.clear()

    web._web_cache["hello"] = ([{"rank": 1, "title": "Cached", "snippet": "cached", "url": "https://example.com"}], 100.0)

    sleep_calls: list[float] = []
    monkeypatch.setattr(web.time, "time", lambda: 100.5)
    monkeypatch.setattr(web.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(web, "DDGS", lambda: (_ for _ in ()).throw(AssertionError("DDGS should not be called for cache hits")))

    result = web._web_search(ctx, web.WebSearchArgs(query="hello"))

    assert result["cached"] is True
    assert result["results"][0]["title"] == "Cached"
    assert sleep_calls == []


def test_web_search_live_repeat_query_throttles_only_live_fetches(tmp_path, monkeypatch) -> None:
    ctx = _ctx(tmp_path, monkeypatch)
    web._web_cache.clear()
    web._query_last_call.clear()

    clock = {"value": 100.0}
    sleep_calls: list[float] = []

    def fake_time() -> float:
        return clock["value"]

    monkeypatch.setattr(web.time, "time", fake_time)
    monkeypatch.setattr(web.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(
        web,
        "DDGS",
        lambda: _FakeDDGS(
            results=[
                {"href": "https://example.com/a", "title": "A", "body": "snippet a"},
                {"href": "https://example.com/b", "title": "B", "body": "snippet b"},
            ]
        ),
    )

    first = web._web_search(ctx, web.WebSearchArgs(query="hello"))
    assert first["cached"] is False
    assert sleep_calls == []

    web._web_cache.clear()
    clock["value"] = 100.5

    second = web._web_search(ctx, web.WebSearchArgs(query="hello"))
    assert second["cached"] is False
    assert sleep_calls == [1.0]
