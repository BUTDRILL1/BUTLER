from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from butler.sandbox import SandboxError
from butler.tools.base import Tool, ToolContext, ToolError


class ListArgs(BaseModel):
    path: str = Field(default=".")
    max_entries: int = Field(default=200, ge=1, le=5000)


def _list_dir(ctx: ToolContext, args: ListArgs) -> dict[str, Any]:
    if not ctx.sandbox.allowed_roots:
        raise ToolError("No allowed roots configured. Add one with /roots add <path>.")

    requested = Path(args.path)
    resolved = ctx.sandbox.ensure_allowed(requested if requested.is_absolute() else (ctx.sandbox.allowed_roots[0] / requested))
    if not resolved.exists():
        raise ToolError(f"Not found: {resolved}")
    if not resolved.is_dir():
        raise ToolError(f"Not a directory: {resolved}")

    entries: list[dict[str, Any]] = []
    for i, child in enumerate(sorted(resolved.iterdir(), key=lambda p: p.name.lower())):
        if i >= args.max_entries:
            break
        try:
            child_res = ctx.sandbox.ensure_allowed(child)
        except SandboxError:
            continue
        entries.append(
            {
                "name": child_res.name,
                "path": str(child_res),
                "is_dir": child_res.is_dir(),
                "size_bytes": child_res.stat().st_size if child_res.is_file() else None,
            }
        )
    return {"path": str(resolved), "entries": entries}


class ReadTextArgs(BaseModel):
    path: str


def _read_text(ctx: ToolContext, args: ReadTextArgs) -> dict[str, Any]:
    if not ctx.sandbox.allowed_roots:
        raise ToolError("No allowed roots configured. Add one with /roots add <path>.")
    p = ctx.sandbox.ensure_allowed(Path(args.path))
    if not p.exists() or not p.is_file():
        raise ToolError(f"Not a file: {p}")
    size = p.stat().st_size
    if size > ctx.config.max_file_bytes:
        raise ToolError(f"File too large ({size} bytes). Limit is {ctx.config.max_file_bytes}.")
    content = p.read_text(encoding="utf-8", errors="replace")
    return {"path": str(p), "size_bytes": size, "content": content}


class SearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=400)
    # Allow higher inputs but clamp internally (Gemma-friendly).
    limit: int = Field(default=10, ge=1, le=10_000)


def _clean_snippet(s: str) -> str:
    return s.strip().replace("\n", " ")


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip()


def _is_invalid_fts_query_error(e: sqlite3.OperationalError) -> bool:
    msg = str(e).lower()
    return ("fts" in msg and "syntax" in msg) or ("syntax error" in msg) or ("parse" in msg)


def _is_bm25_missing_error(e: sqlite3.OperationalError) -> bool:
    msg = str(e).lower()
    return "bm25" in msg and ("no such function" in msg or "unknown function" in msg or "bm25" in msg)


def _files_search(ctx: ToolContext, args: SearchArgs) -> dict[str, Any]:
    # Hard cap regardless of caller input (Gemma-friendly).
    limit = max(1, min(int(args.limit), 10))

    row = ctx.db.conn.execute("SELECT 1 FROM files_fts LIMIT 1").fetchone()
    if row is None:
        raise ToolError("Index is empty. Run /index sync first.")

    query = args.query.strip()

    def format_rows(rows: list[sqlite3.Row], *, score_key: str | None) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for r in rows:
            snippet = _truncate(r["snippet"] or "", 200)
            # Keep snippets clean for small local models.
            snippet = _clean_snippet(snippet)
            snippet = _truncate(snippet, 200)
            results.append(
                {
                    "path": r["path"],
                    "score": (float(r[score_key]) if score_key and r[score_key] is not None else None),
                    "snippet": snippet,
                }
            )
        # Deduplicate paths while preserving order.
        seen: set[str] = set()
        results = [r for r in results if not (r["path"] in seen or seen.add(r["path"]))]
        return results[:limit]

    # Try bm25 ranking first (stable tie-breaker on path).
    try:
        rows = ctx.db.conn.execute(
            """
            SELECT
              path,
              bm25(files_fts) AS score,
              snippet(files_fts, 1, '', '', ' … ', 8) AS snippet
            FROM files_fts
            WHERE files_fts MATCH ?
            ORDER BY score, path
            LIMIT ?
            """,
            (query, limit * 3),
        ).fetchall()
        return {"query": query, "results": format_rows(rows, score_key="score")}
    except sqlite3.OperationalError as e:
        if _is_invalid_fts_query_error(e):
            raise ToolError(
                "Invalid search query (SQLite FTS syntax). Try simpler keywords, e.g. `error login`."
            ) from e
        if not _is_bm25_missing_error(e):
            raise ToolError(str(e)) from e

    # Fallback without bm25: stable ordering by path.
    try:
        rows = ctx.db.conn.execute(
            """
            SELECT
              path,
              snippet(files_fts, 1, '', '', ' … ', 8) AS snippet
            FROM files_fts
            WHERE files_fts MATCH ?
            ORDER BY path
            LIMIT ?
            """,
            (query, limit * 3),
        ).fetchall()
        return {"query": query, "results": format_rows(rows, score_key=None)}
    except sqlite3.OperationalError as e:
        if _is_invalid_fts_query_error(e):
            raise ToolError(
                "Invalid search query (SQLite FTS syntax). Try simpler keywords, e.g. `error login`."
            ) from e
        raise ToolError(str(e)) from e


def build() -> list[Tool]:
    return [
        Tool(
            name="files.list",
            description="List directory entries within allowed roots.",
            input_model=ListArgs,
            handler=_list_dir,
        ),
        Tool(
            name="files.read_text",
            description="Read a UTF-8 text file within allowed roots (size-capped).",
            input_model=ReadTextArgs,
            handler=_read_text,
        ),
        Tool(
            name="files.search",
            description="Search indexed .md/.txt content (requires /index sync). Returns <=10 short snippets.",
            input_model=SearchArgs,
            handler=_files_search,
            side_effect=False,
        ),
    ]
