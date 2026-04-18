from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from butler.config import ButlerConfig
from butler.paths import db_path, ensure_dir, notes_dir


SCHEMA_VERSION = 1


def _connect(path: Path) -> sqlite3.Connection:
    ensure_dir(path.parent)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _apply_migrations(conn: sqlite3.Connection) -> None:
    current = conn.execute("PRAGMA user_version;").fetchone()[0]
    if current >= SCHEMA_VERSION:
        return

    if current < 1:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
              id TEXT PRIMARY KEY,
              created_at_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
              id TEXT PRIMARY KEY,
              conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
              role TEXT NOT NULL,
              content TEXT NOT NULL,
              created_at_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tool_calls (
              id TEXT PRIMARY KEY,
              conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
              tool_name TEXT NOT NULL,
              args_json TEXT NOT NULL,
              result_json TEXT,
              status TEXT NOT NULL,
              error TEXT,
              started_at_ms INTEGER NOT NULL,
              duration_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS notes (
              id TEXT PRIMARY KEY,
              title TEXT NOT NULL UNIQUE,
              path TEXT NOT NULL UNIQUE,
              created_at_ms INTEGER NOT NULL,
              updated_at_ms INTEGER NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(title, content, path);
            CREATE TABLE IF NOT EXISTS roots (
              path TEXT PRIMARY KEY,
              created_at_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS indexed_files (
              path TEXT PRIMARY KEY,
              size_bytes INTEGER NOT NULL,
              mtime_ms INTEGER NOT NULL,
              indexed_at_ms INTEGER NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(path, content);
            """
        )
        conn.execute("PRAGMA user_version = 1;")
        conn.commit()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _uuid() -> str:
    import uuid

    return str(uuid.uuid4())


@dataclass
class ButlerDB:
    conn: sqlite3.Connection

    @staticmethod
    def open(config: ButlerConfig) -> "ButlerDB":
        # Ensure home folders exist as a side effect.
        ensure_dir(db_path().parent)
        ensure_dir(notes_dir())
        conn = _connect(db_path())
        _apply_migrations(conn)
        return ButlerDB(conn=conn)

    def new_conversation(self) -> str:
        cid = _uuid()
        self.conn.execute(
            "INSERT INTO conversations (id, created_at_ms) VALUES (?, ?)",
            (cid, _now_ms()),
        )
        self.conn.commit()
        return cid

    def get_last_conversation(self, max_age_hours: int = 24) -> str | None:
        cutoff = _now_ms() - (max_age_hours * 3600 * 1000)
        row = self.conn.execute(
            "SELECT id FROM conversations WHERE created_at_ms > ? ORDER BY created_at_ms DESC LIMIT 1",
            (cutoff,)
        ).fetchone()
        return row["id"] if row else None

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at_ms) VALUES (?, ?, ?, ?, ?)",
            (_uuid(), conversation_id, role, content, _now_ms()),
        )
        self.conn.commit()

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY created_at_ms ASC",
            (conversation_id,),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def log_tool_call(
        self,
        conversation_id: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        started_at_ms: int,
        duration_ms: int,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO tool_calls
              (id, conversation_id, tool_name, args_json, result_json, status, error, started_at_ms, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _uuid(),
                conversation_id,
                tool_name,
                json.dumps(args, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False) if result is not None else None,
                status,
                error,
                started_at_ms,
                duration_ms,
            ),
        )
        self.conn.commit()
