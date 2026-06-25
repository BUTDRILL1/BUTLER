from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from butler.tools.base import Tool, ToolContext


def build() -> list[Tool]:
    return [
        Tool(
            name="reminders.create",
            description=(
                "Schedule a proactive reminder for the user. "
                "The reminder will be pushed to the user's phone via Telegram at the specified time. "
                "Provide the reminder message and the number of minutes from now to trigger it. "
                "If the user wants a recurring reminder (e.g. 'every 30 minutes'), also provide recurrence_minutes."
            ),
            input_model=CreateReminderArgs,
            side_effect=True,
            handler=create_reminder,
        ),
        Tool(
            name="reminders.list",
            description="List all currently active (pending) reminders that are scheduled for the future.",
            input_model=ListRemindersArgs,
            side_effect=False,
            handler=list_reminders,
        ),
        Tool(
            name="reminders.delete",
            description="Delete a specific pending reminder. You can provide either the exact reminder ID, or a fuzzy text query (e.g., 'water') to match against the message.",
            input_model=DeleteReminderArgs,
            side_effect=True,
            handler=delete_reminder,
        ),
        Tool(
            name="reminders.clear",
            description="Clear and delete ALL pending reminders.",
            input_model=ClearRemindersArgs,
            side_effect=True,
            handler=clear_reminders,
        ),
    ]


class CreateReminderArgs(BaseModel):
    message: str = Field(description="The message to remind the user about (e.g. 'Drink water')")
    minutes_from_now: float = Field(description="How many minutes from right now the first reminder should trigger.")
    recurrence_minutes: float | None = Field(
        default=None,
        description="If set, the reminder will repeat every this many minutes after each firing. Leave None for one-shot reminders.",
    )


def create_reminder(ctx: ToolContext, args: CreateReminderArgs) -> dict[str, Any]:
    if args.minutes_from_now <= 0:
        return {"error": "minutes_from_now must be greater than zero."}

    trigger_time_ms = int(time.time() * 1000) + int(args.minutes_from_now * 60 * 1000)
    rid = ctx.db.create_reminder(args.message, trigger_time_ms, recurrence_minutes=args.recurrence_minutes)

    result: dict[str, Any] = {
        "status": "success",
        "reminder_id": rid,
        "message": args.message,
        "trigger_time_ms": trigger_time_ms,
        "scheduled_in_minutes": args.minutes_from_now,
    }
    if args.recurrence_minutes:
        result["recurring_every_minutes"] = args.recurrence_minutes

    return result


class ListRemindersArgs(BaseModel):
    pass


def list_reminders(ctx: ToolContext, args: ListRemindersArgs) -> dict[str, Any]:
    from datetime import datetime
    reminders = ctx.db.list_all_pending_reminders()
    now_ms = int(time.time() * 1000)

    formatted = []
    for r in reminders:
        minutes_left = max(0.0, (r["trigger_time_ms"] - now_ms) / 1000.0 / 60.0)
        dt = datetime.fromtimestamp(r["trigger_time_ms"] / 1000.0)
        time_str = dt.strftime("%A at %I:%M %p")
        
        entry: dict[str, Any] = {
            "id": r["id"],
            "message": r["message"],
            "scheduled_time": f"{time_str} (in {round(minutes_left, 1)} minutes)",
        }
        if r.get("recurrence_minutes"):
            entry["recurring_every_minutes"] = r["recurrence_minutes"]
        formatted.append(entry)

    return {"count": len(formatted), "reminders": formatted}


class DeleteReminderArgs(BaseModel):
    query: str = Field(description="The exact ID of the reminder, or a keyword to search for in the message (e.g., 'water').")

def delete_reminder(ctx: ToolContext, args: DeleteReminderArgs) -> dict[str, Any]:
    reminders = ctx.db.list_all_pending_reminders()
    
    # Try exact ID match first
    target_id = None
    for r in reminders:
        if r["id"] == args.query:
            target_id = r["id"]
            break
            
    # Fallback to fuzzy message match
    if not target_id:
        query_lower = args.query.lower()
        for r in reminders:
            if query_lower in r["message"].lower():
                target_id = r["id"]
                break
                
    if not target_id:
        return {"error": f"No active reminder found matching '{args.query}'"}
        
    deleted = ctx.db.delete_reminder(target_id)
    return {"status": "success", "deleted": deleted, "reminder_id": target_id}

class ClearRemindersArgs(BaseModel):
    pass

def clear_reminders(ctx: ToolContext, args: ClearRemindersArgs) -> dict[str, Any]:
    count = ctx.db.clear_reminders()
    return {"status": "success", "cleared_count": count}
