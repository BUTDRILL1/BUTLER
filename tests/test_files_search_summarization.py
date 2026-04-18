from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

import pytest

sys.path.insert(0, "src")

from butler.agent.loop import AgentRuntime  # noqa: E402
from butler.config import ButlerConfig  # noqa: E402
from butler.db import ButlerDB  # noqa: E402
from butler.tools.registry import build_default_tool_registry  # noqa: E402


@dataclass
class FakeProvider:
    responses: list[str]

    def chat(self, messages: list[dict[str, Any]], *, temperature: float = 0.2) -> str:
        if not self.responses:
            return '{"type":"final","content":"(no more scripted responses)"}'
        return self.responses.pop(0)


def _seed_files_fts(db: ButlerDB) -> None:
    db.conn.execute("INSERT INTO files_fts (path, content) VALUES (?, ?)", ("C:/a.txt", "hello world " * 5))
    db.conn.execute("INSERT INTO files_fts (path, content) VALUES (?, ?)", ("C:/b.txt", "hello there " * 5))
    db.conn.execute("INSERT INTO files_fts (path, content) VALUES (?, ?)", ("C:/c.txt", "hello again " * 5))
    db.conn.commit()


def test_files_search_summarization_normal_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BUTLER_HOME", str(tmp_path))
    cfg = ButlerConfig()
    db = ButlerDB.open(cfg)
    _seed_files_fts(db)
    tools = build_default_tool_registry(cfg, db)

    provider = FakeProvider(
        responses=[
            '{"type":"tool_call","name":"files.search","arguments":{"query":"hello","limit":10}}',
            '{"type":"final","content":"Best match:\\nC:/a.txt — hello world\\n\\nOther matches:\\n1. C:/b.txt — hello there\\n2. C:/c.txt — hello again"}',
        ]
    )
    runtime = AgentRuntime(config=cfg, db=db, tools=tools, provider=provider)
    out = runtime.chat_once("find hello")
    assert out.startswith("Best match:")


def test_files_search_summarization_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BUTLER_HOME", str(tmp_path))
    cfg = ButlerConfig()
    db = ButlerDB.open(cfg)
    _seed_files_fts(db)
    tools = build_default_tool_registry(cfg, db)

    provider = FakeProvider(
        responses=[
            '{"type":"tool_call","name":"files.search","arguments":{"query":"hello","limit":10}}',
            "not json at all",
            "still not json",
            "also not json",
        ]
    )
    runtime = AgentRuntime(config=cfg, db=db, tools=tools, provider=provider)
    out = runtime.chat_once("find hello")
    assert out.startswith("Best match:")
    assert "Other matches:" in out
