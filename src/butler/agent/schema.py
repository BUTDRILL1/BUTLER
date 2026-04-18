from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCallAction(BaseModel):
    type: Literal["tool_call"]
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ClarifyAction(BaseModel):
    type: Literal["clarify"]
    question: str = Field(min_length=1)
    choices: list[str] | None = None


class FinalAction(BaseModel):
    type: Literal["final"]
    content: str = Field(min_length=1)


Action = ToolCallAction | ClarifyAction | FinalAction
