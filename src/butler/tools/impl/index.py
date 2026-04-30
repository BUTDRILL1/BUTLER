from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from butler.tools.base import Tool, ToolContext, ToolError


class SyncArgs(BaseModel):
    max_files: int = Field(default=2000, ge=1, le=20000)
    paths: list[str] | None = Field(default=None, description="Optional specific paths to sync incrementally.")


TEXT_EXTS = {".md", ".txt"}
PDF_EXTS = {".pdf"}


def _read_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover - dependency missing is handled at runtime
        raise ToolError("PDF support requires the 'pypdf' package.") from e

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n".join(pages).strip()


def _extract_index_text(path: Path) -> tuple[str | None, bool]:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTS:
        return path.read_text(encoding="utf-8", errors="replace"), False
    if suffix in PDF_EXTS:
        return _read_pdf_text(path), True
    return None, False


def _index_sync(ctx: ToolContext, args: SyncArgs) -> dict[str, Any]:
    if not ctx.sandbox.allowed_roots:
        raise ToolError("No allowed roots configured. Add one with /roots add <path>.")

    started = time.time()
    scanned = 0
    indexed = 0
    skipped_large = 0
    skipped_binary = 0

    if args.paths:
        # Incremental Sync
        targets = [Path(p) for p in args.paths]
    else:
        # Full Sync
        ctx.db.conn.execute("DELETE FROM files_fts;")
        # For memory, we don't wipe everything as it might contain chat history, 
        # but for simplicity in v1 full sync we just scan all.
        targets = []
        for root in ctx.sandbox.allowed_roots:
            targets.extend(list(root.rglob("*")))

    for path in targets:
        if indexed >= args.max_files:
            break
        if not path.is_file():
            continue
        scanned += 1
        if path.suffix.lower() not in TEXT_EXTS | PDF_EXTS:
            skipped_binary += 1
            continue
        size = path.stat().st_size
        if size > ctx.config.max_file_bytes:
            skipped_large += 1
            continue
            
        content, is_pdf = _extract_index_text(path)
        if content is None:
            skipped_binary += 1
            continue
            
        if is_pdf and not content.strip():
            content = f"{path.name}"
            
        # Clear existing entries for this path to avoid duplicates
        ctx.db.conn.execute("DELETE FROM files_fts WHERE path = ?", (str(path),))
        # Note: Ideally we would also clear memory by metadata, but memory is additive in this v1.
        
        ctx.db.conn.execute(
            "INSERT INTO files_fts (path, content) VALUES (?, ?)",
            (str(path), content),
        )
        
        if ctx.memory:
            chunks = [content[i:i+1000] for i in range(0, len(content), 800)]
            for i, chunk in enumerate(chunks):
                ctx.memory.add(
                    chunk, 
                    metadata={"path": str(path), "type": "file", "chunk_idx": i}
                )
        
        indexed += 1

    ctx.db.conn.commit()
    elapsed_ms = int((time.time() - started) * 1000)
    return {
        "mode": "incremental" if args.paths else "full",
        "scanned": scanned,
        "indexed": indexed,
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
