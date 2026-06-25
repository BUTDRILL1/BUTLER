from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from supabase import create_client, Client

from butler.config import ButlerConfig


def _now_ms() -> int:
    return int(time.time() * 1000)


def _uuid() -> str:
    return str(uuid.uuid4())


@dataclass
class ButlerDB:
    client: Client

    @staticmethod
    def open(config: ButlerConfig) -> "ButlerDB":
        url: str = config.supabase_url
        key: str = config.supabase_key
        
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in the config/.env")
            
        client = create_client(url, key)
        db = ButlerDB(client=client)
        db.cleanup_old_conversations()
        return db

    def cleanup_old_conversations(self, max_age_days: int = 14) -> None:
        cutoff = _now_ms() - (max_age_days * 24 * 3600 * 1000)
        self.client.table("conversations").delete().lt("created_at_ms", cutoff).execute()

    def new_conversation(self) -> str:
        cid = _uuid()
        self.client.table("conversations").insert({
            "id": cid, 
            "created_at_ms": _now_ms()
        }).execute()
        return cid

    def get_last_conversation(self, max_age_hours: int = 24) -> str | None:
        cutoff = _now_ms() - (max_age_hours * 3600 * 1000)
        res = self.client.table("conversations") \
            .select("id") \
            .gt("created_at_ms", cutoff) \
            .order("created_at_ms", desc=True) \
            .limit(1) \
            .execute()
        return res.data[0]["id"] if res.data else None

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        self.client.table("messages").insert({
            "id": _uuid(),
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "created_at_ms": _now_ms()
        }).execute()

    def list_messages(self, conversation_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        query = self.client.table("messages") \
            .select("role, content") \
            .eq("conversation_id", conversation_id) \
            .order("created_at_ms", desc=False)
        
        if limit:
            # We want the LAST `limit` messages in ascending order.
            # Supabase doesn't easily let us order desc, limit, then reverse in one query.
            # So we'll fetch them desc, then reverse in Python.
            query = self.client.table("messages") \
                .select("role, content") \
                .eq("conversation_id", conversation_id) \
                .order("created_at_ms", desc=True) \
                .limit(limit)
            res = query.execute()
            return [{"role": r["role"], "content": r["content"]} for r in reversed(res.data)]

        res = query.execute()
        return [{"role": r["role"], "content": r["content"]} for r in res.data]

    def clear_conversation(self, conversation_id: str) -> None:
        self.client.table("messages").delete().eq("conversation_id", conversation_id).execute()

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
        self.client.table("tool_calls").insert({
            "id": _uuid(),
            "conversation_id": conversation_id,
            "tool_name": tool_name,
            "args_json": json.dumps(args, ensure_ascii=False),
            "result_json": json.dumps(result, ensure_ascii=False) if result is not None else None,
            "status": status,
            "error": error,
            "started_at_ms": started_at_ms,
            "duration_ms": duration_ms
        }).execute()

    def flush_turn(self) -> None:
        pass  # Supabase inserts are atomic and immediately executed.

    # --- Reminders API ---

    def create_reminder(self, message: str, trigger_time_ms: int, recurrence_minutes: float | None = None) -> str:
        rid = _uuid()
        self.client.table("reminders").insert({
            "id": rid,
            "message": message,
            "trigger_time_ms": trigger_time_ms,
            "recurrence_minutes": recurrence_minutes,
            "is_sent": 0,
            "created_at_ms": _now_ms()
        }).execute()
        return rid

    def get_pending_reminders(self) -> list[dict[str, Any]]:
        now = _now_ms()
        res = self.client.table("reminders") \
            .select("id, message, trigger_time_ms, recurrence_minutes") \
            .eq("is_sent", 0) \
            .lte("trigger_time_ms", now) \
            .order("trigger_time_ms", desc=False) \
            .execute()
        return res.data

    def mark_reminder_sent(self, reminder_id: str) -> None:
        self.client.table("reminders").update({"is_sent": 1}).eq("id", reminder_id).execute()

    def snooze_reminder(self, reminder_id: str, added_time_ms: int) -> None:
        # We must fetch the current trigger_time_ms, then update it.
        res = self.client.table("reminders").select("trigger_time_ms").eq("id", reminder_id).execute()
        if res.data:
            current_ms = res.data[0]["trigger_time_ms"]
            self.client.table("reminders").update({
                "is_sent": 0, 
                "trigger_time_ms": current_ms + added_time_ms
            }).eq("id", reminder_id).execute()

    def reschedule_recurring_reminder(self, reminder_id: str, recurrence_minutes: float) -> None:
        added_ms = int(recurrence_minutes * 60 * 1000)
        res = self.client.table("reminders").select("trigger_time_ms").eq("id", reminder_id).execute()
        if res.data:
            current_ms = res.data[0]["trigger_time_ms"]
            self.client.table("reminders").update({
                "trigger_time_ms": current_ms + added_ms
            }).eq("id", reminder_id).execute()

    def list_all_pending_reminders(self) -> list[dict[str, Any]]:
        res = self.client.table("reminders") \
            .select("id, message, trigger_time_ms, recurrence_minutes") \
            .eq("is_sent", 0) \
            .order("trigger_time_ms", desc=False) \
            .execute()
        return res.data

    def delete_reminder(self, reminder_id: str) -> bool:
        res = self.client.table("reminders").delete().eq("id", reminder_id).execute()
        return len(res.data) > 0 if res.data else False

    def clear_reminders(self) -> int:
        res = self.client.table("reminders").delete().eq("is_sent", 0).execute()
        return len(res.data) if res.data else 0

    # ── Custom Skills ──────────────────────────────────────────────────

    def add_skill(self, trigger: str, action: str) -> None:
        self.client.table("skills").upsert({
            "id": _uuid(),
            "trigger": trigger,
            "action": action,
            "created_at_ms": _now_ms()
        }, on_conflict="trigger").execute()

    def list_skills(self) -> list[dict[str, Any]]:
        res = self.client.table("skills").select("id, trigger, action").order("created_at_ms", desc=True).execute()
        return res.data

    def delete_skill(self, skill_id: str) -> bool:
        res = self.client.table("skills").delete().eq("id", skill_id).execute()
        return len(res.data) > 0 if res.data else False
