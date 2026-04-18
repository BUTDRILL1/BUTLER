from pathlib import Path

from butler.config import ButlerConfig
from butler.db import ButlerDB
from butler.sandbox import PathSandbox
from butler.tools.base import ToolContext
from butler.tools.impl.notes import CreateArgs, ReadArgs, _notes_create, _notes_read


def test_notes_create_and_read(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BUTLER_HOME", str(tmp_path))
    cfg = ButlerConfig()
    db = ButlerDB.open(cfg)
    ctx = ToolContext(config=cfg, db=db, sandbox=PathSandbox.from_strings([]))

    out = _notes_create(ctx, CreateArgs(title="Test Note", content="hello"))
    assert Path(out["path"]).exists()

    read = _notes_read(ctx, ReadArgs(title="Test Note"))
    assert "hello" in read["content"]
