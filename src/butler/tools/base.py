from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from butler.config import ButlerConfig
from butler.db import ButlerDB
from butler.sandbox import PathSandbox


class ToolError(Exception):
    pass


InputT = TypeVar("InputT", bound=BaseModel)


@dataclass(frozen=True)
class ToolContext:
    config: ButlerConfig
    db: ButlerDB
    sandbox: PathSandbox


@dataclass(frozen=True)
class Tool(Generic[InputT]):
    name: str
    description: str
    input_model: type[InputT]
    handler: Callable[[ToolContext, InputT], dict[str, Any]]
    side_effect: bool = False

    def call(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = self.input_model.model_validate(args)
        except ValidationError as e:
            raise ToolError(f"Invalid arguments for {self.name}: {e}") from e
        return self.handler(ctx, parsed)


@dataclass(frozen=True)
class ToolCallRecord:
    tool_name: str
    args: dict[str, Any]
    started_at_ms: int
    duration_ms: int
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None


def now_ms() -> int:
    return int(time.time() * 1000)
