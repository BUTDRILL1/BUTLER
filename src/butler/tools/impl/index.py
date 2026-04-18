from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from butler.tools.base import Tool, ToolContext, ToolError


class SyncArgs(BaseModel):
    max_files: int = Field(default=2000, ge=1, le=20000)


TEXT_EXTS = {".md", ".txt"}


def _index_sync(ctx: ToolContext, args: SyncArgs) -> dict[str, Any]:
    if not ctx.sandbox.allowed_roots:
        raise ToolError("No allowed roots configured. Add one with /roots add <path>.")

    started = time.time()
    scanned = 0
    indexed = 0
    skipped_large = 0
    skipped_binary = 0

    # A simple full rebuild for v1 (deterministic and easy to reason about).
    ctx.db.conn.execute("DELETE FROM files_fts;")

    for root in ctx.sandbox.allowed_roots:
        for path in root.rglob("*"):
            if indexed >= args.max_files:
                break
            if not path.is_file():
                continue
            scanned += 1
            if path.suffix.lower() not in TEXT_EXTS:
                skipped_binary += 1
                continue
            size = path.stat().st_size
            if size > ctx.config.max_file_bytes:
                skipped_large += 1
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            ctx.db.conn.execute(
                "INSERT INTO files_fts (path, content) VALUES (?, ?)",
                (str(path), content),
            )
            indexed += 1

    ctx.db.conn.commit()
    elapsed_ms = int((time.time() - started) * 1000)
    return {
        "roots": [str(r) for r in ctx.sandbox.allowed_roots],
        "scanned": scanned,
        "indexed": indexed,
        "skipped_large": skipped_large,
        "skipped_non_text": skipped_binary,
        "elapsed_ms": elapsed_ms,
    }


def build() -> list[Tool]:
    return [
        Tool(
            name="index.sync",
            description="Index .md/.txt files under allowed roots into SQLite FTS.",
            input_model=SyncArgs,
            handler=_index_sync,
            side_effect=True,
        )
    ]
