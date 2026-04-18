from __future__ import annotations

import platform
import time
from typing import Any

from pydantic import BaseModel

from butler.tools.base import Tool, ToolContext


class NowArgs(BaseModel):
    pass


def _now(ctx: ToolContext, args: NowArgs) -> dict[str, Any]:
    return {
        "epoch_ms": int(time.time() * 1000),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def build() -> list[Tool]:
    return [
        Tool(
            name="system.now",
            description="Get current time and basic runtime info.",
            input_model=NowArgs,
            handler=_now,
            side_effect=False,
        )
    ]
