from __future__ import annotations

from pathlib import Path

import pytest

from butler.config import ButlerConfig
from butler.db import ButlerDB
from butler.sandbox import PathSandbox
from butler.tools.base import ToolContext, ToolError
from butler.tools.impl.files import SearchArgs, _files_search


def test_files_search_limits_snippet_and_dedups(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BUTLER_HOME", str(tmp_path))
    cfg = ButlerConfig()
    db = ButlerDB.open(cfg)
    ctx = ToolContext(config=cfg, db=db, sandbox=PathSandbox.from_strings([]))

    # Seed FTS with duplicates and newlines.
    db.conn.execute("INSERT INTO files_fts (path, content) VALUES (?, ?)", ("C:\\a.txt", "hello\nworld " * 50))
    db.conn.execute("INSERT INTO files_fts (path, content) VALUES (?, ?)", ("C:\\a.txt", "hello again"))
    db.conn.execute("INSERT INTO files_fts (path, content) VALUES (?, ?)", ("C:\\b.txt", "hello there"))
    db.conn.commit()

    out = _files_search(ctx, SearchArgs(query="hello", limit=999))
    results = out["results"]
    assert len(results) <= 10
    assert len({r["path"] for r in results}) == len(results)
    for r in results:
        assert "\n" not in r["snippet"]
        assert len(r["snippet"]) <= 200


def test_files_search_empty_index_errors(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BUTLER_HOME", str(tmp_path))
    cfg = ButlerConfig()
    db = ButlerDB.open(cfg)
    ctx = ToolContext(config=cfg, db=db, sandbox=PathSandbox.from_strings([]))

    with pytest.raises(ToolError):
        _files_search(ctx, SearchArgs(query="anything", limit=10))


def test_files_search_invalid_query_is_friendly(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BUTLER_HOME", str(tmp_path))
    cfg = ButlerConfig()
    db = ButlerDB.open(cfg)
    ctx = ToolContext(config=cfg, db=db, sandbox=PathSandbox.from_strings([]))

    db.conn.execute("INSERT INTO files_fts (path, content) VALUES (?, ?)", ("C:\\a.txt", "hello world"))
    db.conn.commit()

    with pytest.raises(ToolError):
        _files_search(ctx, SearchArgs(query='"', limit=10))

