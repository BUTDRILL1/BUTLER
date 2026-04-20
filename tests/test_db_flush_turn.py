from __future__ import annotations

import sqlite3

from butler.config import ButlerConfig
from butler.db import ButlerDB
from butler.paths import db_path


def test_flush_turn_commits_batched_writes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BUTLER_HOME", str(tmp_path))
    cfg = ButlerConfig()
    db = ButlerDB.open(cfg)

    journal_mode = db.conn.execute("PRAGMA journal_mode;").fetchone()[0]
    assert str(journal_mode).lower() == "wal"

    conversation_id = db.new_conversation()
    db.add_message(conversation_id, "user", "hello")

    raw = sqlite3.connect(str(db_path()))
    raw.row_factory = sqlite3.Row
    try:
        before = raw.execute(
            "SELECT count(*) AS count FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()["count"]
        assert before == 0

        db.flush_turn()

        after = raw.execute(
            "SELECT count(*) AS count FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()["count"]
        assert after == 1
    finally:
        raw.close()


def test_list_messages_limit_returns_latest_rows_in_order(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BUTLER_HOME", str(tmp_path))
    cfg = ButlerConfig()
    db = ButlerDB.open(cfg)

    conversation_id = db.new_conversation()
    db.add_message(conversation_id, "user", "one")
    db.add_message(conversation_id, "assistant", "two")
    db.add_message(conversation_id, "user", "three")
    db.flush_turn()

    messages = db.list_messages(conversation_id, limit=2)
    assert [m["content"] for m in messages] == ["two", "three"]
