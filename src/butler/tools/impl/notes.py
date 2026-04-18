from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from butler.paths import ensure_dir, notes_dir
from butler.tools.base import Tool, ToolContext, ToolError


def _slugify(title: str) -> str:
    t = title.strip().lower()
    t = re.sub(r"[^a-z0-9]+", "-", t)
    t = re.sub(r"-{2,}", "-", t).strip("-")
    return t or "note"


def _note_path_for(title: str) -> Path:
    slug = _slugify(title)
    return (notes_dir() / f"{slug}.md").resolve()


class CreateArgs(BaseModel):
    title: str = Field(min_length=1, max_length=80)
    content: str = Field(default="")


def _notes_create(ctx: ToolContext, args: CreateArgs) -> dict[str, Any]:
    now = int(time.time() * 1000)
    p = _note_path_for(args.title)
    ensure_dir(p.parent)
    if p.exists():
        raise ToolError("Note already exists (by slug). Use notes.append or choose a different title.")
    p.write_text(args.content.strip() + "\n", encoding="utf-8")

    ctx.db.conn.execute(
        "INSERT INTO notes (id, title, path, created_at_ms, updated_at_ms) VALUES (?, ?, ?, ?, ?)",
        (_uuid(), args.title.strip(), str(p), now, now),
    )
    ctx.db.conn.execute(
        "INSERT INTO notes_fts (title, content, path) VALUES (?, ?, ?)",
        (args.title.strip(), args.content, str(p)),
    )
    ctx.db.conn.commit()
    return {"title": args.title.strip(), "path": str(p)}


class AppendArgs(BaseModel):
    title: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1)


def _notes_append(ctx: ToolContext, args: AppendArgs) -> dict[str, Any]:
    row = ctx.db.conn.execute("SELECT path FROM notes WHERE title=?", (args.title.strip(),)).fetchone()
    if not row:
        raise ToolError("Note not found. Create it first.")
    p = Path(row["path"])
    p.write_text(p.read_text(encoding="utf-8", errors="replace") + args.content.strip() + "\n", encoding="utf-8")

    updated = int(time.time() * 1000)
    content = p.read_text(encoding="utf-8", errors="replace")
    ctx.db.conn.execute("UPDATE notes SET updated_at_ms=? WHERE title=?", (updated, args.title.strip()))
    ctx.db.conn.execute("DELETE FROM notes_fts WHERE path=?", (str(p),))
    ctx.db.conn.execute("INSERT INTO notes_fts (title, content, path) VALUES (?, ?, ?)", (args.title.strip(), content, str(p)))
    ctx.db.conn.commit()
    return {"title": args.title.strip(), "path": str(p), "updated_at_ms": updated}


class ReadArgs(BaseModel):
    title: str = Field(min_length=1, max_length=80)


def _notes_read(ctx: ToolContext, args: ReadArgs) -> dict[str, Any]:
    row = ctx.db.conn.execute("SELECT path, updated_at_ms FROM notes WHERE title=?", (args.title.strip(),)).fetchone()
    if not row:
        raise ToolError("Note not found.")
    p = Path(row["path"])
    content = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
    return {"title": args.title.strip(), "path": str(p), "updated_at_ms": row["updated_at_ms"], "content": content}


class SearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=10, ge=1, le=50)


def _notes_search(ctx: ToolContext, args: SearchArgs) -> dict[str, Any]:
    rows = ctx.db.conn.execute(
        "SELECT title, path FROM notes_fts WHERE notes_fts MATCH ? LIMIT ?",
        (args.query, args.limit),
    ).fetchall()
    return {"query": args.query, "results": [{"title": r["title"], "path": r["path"]} for r in rows]}


def _uuid() -> str:
    import uuid

    return str(uuid.uuid4())


def build() -> list[Tool]:
    return [
        Tool(
            name="notes.create",
            description="Create a new markdown note in BUTLER notes directory.",
            input_model=CreateArgs,
            handler=_notes_create,
            side_effect=True,
        ),
        Tool(
            name="notes.append",
            description="Append content to an existing note.",
            input_model=AppendArgs,
            handler=_notes_append,
            side_effect=True,
        ),
        Tool(
            name="notes.read",
            description="Read a note by title.",
            input_model=ReadArgs,
            handler=_notes_read,
        ),
        Tool(
            name="notes.search",
            description="Search notes using SQLite FTS.",
            input_model=SearchArgs,
            handler=_notes_search,
        ),
    ]
