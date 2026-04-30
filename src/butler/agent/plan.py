from __future__ import annotations
from typing import Any, List, Optional
from pydantic import BaseModel, Field

class TaskStep(BaseModel):
    id: int
    tool_name: str
    arguments: dict[str, Any]
    description: str
    narration: str = Field(description="A short, persona-aware status update for TTS.")

class TaskPlan(BaseModel):
    goal: str
    steps: List[TaskStep]
    estimated_duration_seconds: Optional[int] = None

class PlanResult(BaseModel):
    plan: Optional[TaskPlan] = None
    requires_clarification: bool = False
    clarification_question: Optional[str] = None
    is_direct_chat: bool = False
