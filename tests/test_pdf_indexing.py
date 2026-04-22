from __future__ import annotations

from pathlib import Path

from butler.config import ButlerConfig
from butler.db import ButlerDB
from butler.sandbox import PathSandbox
from butler.tools.base import ToolContext
from butler.tools.impl.files import SearchArgs, _files_search
from butler.tools.impl.index import SyncArgs, _index_sync


def test_pdf_files_are_indexed_and_searchable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BUTLER_HOME", str(tmp_path))
    root = tmp_path / "docs"
    root.mkdir()
    pdf_path = root / "SCV_AMIL.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake pdf content")

    cfg = ButlerConfig()
    cfg = cfg.model_copy(update={"allowed_roots": [str(root)]})
    db = ButlerDB.open(cfg)
    sandbox = PathSandbox.from_strings([str(root)])
    ctx = ToolContext(config=cfg, db=db, sandbox=sandbox)

    monkeypatch.setattr("butler.tools.impl.index._read_pdf_text", lambda path: "This resume belongs to Amil Lal.")

    out = _index_sync(ctx, SyncArgs(max_files=20))
    assert out["indexed"] == 1

    search = _files_search(ctx, SearchArgs(query="SCV_AMIL", limit=10))
    assert search["results"]
    assert search["results"][0]["path"].endswith("SCV_AMIL.pdf")

    content_search = _files_search(ctx, SearchArgs(query="Amil Lal", limit=10))
    assert content_search["results"]
    assert content_search["results"][0]["path"].endswith("SCV_AMIL.pdf")
