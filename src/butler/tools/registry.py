from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from butler.config import ButlerConfig
from butler.db import ButlerDB
from butler.sandbox import PathSandbox
from butler.tools.base import Tool, ToolCallRecord, ToolContext, ToolError, now_ms
from butler.tools.impl import files as files_tools
from butler.tools.impl import index as index_tools
from butler.tools.impl import notes as notes_tools
from butler.tools.impl import system as system_tools
from butler.tools.impl import weather as weather_tools
from butler.tools.impl import web as web_tools
from butler.tools.impl import os_control as os_tools
from butler.tools.impl import spotify_control as spotify_tools

logger = logging.getLogger(__name__)


@dataclass
class ToolRegistry:
    config: ButlerConfig
    db: ButlerDB
    sandbox: PathSandbox
    tools: dict[str, Tool]
    _cached_description: list[dict[str, Any]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._cached_description = [
            {
                "name": t.name,
                "description": t.description,
                "side_effect": t.side_effect,
                "parameters": t.input_model.model_json_schema(),
            }
            for t in self.tools.values()
        ]

    def describe(self) -> list[dict[str, Any]]:
        return self._cached_description

    def call(self, tool_name: str, args: dict[str, Any], *, conversation_id: str | None = None) -> dict[str, Any]:
        if tool_name not in self.tools:
            raise ToolError(f"Unknown tool: {tool_name}")
        tool = self.tools[tool_name]
        ctx = ToolContext(config=self.config, db=self.db, sandbox=self.sandbox)

        started = now_ms()
        status = "ok"
        error: str | None = None
        result: dict[str, Any] | None = None
        logger.info("tool_call_start tool=%s conversation_id=%s", tool.name, conversation_id)
        try:
            result = tool.call(ctx, args)
            return result
        except Exception as e:  # noqa: BLE001 - normalize to tool error
            status = "error"
            error = str(e)
            raise
        finally:
            duration = now_ms() - started
            logger.info(
                "tool_call_end tool=%s conversation_id=%s status=%s duration_ms=%d",
                tool.name,
                conversation_id,
                status,
                duration,
            )
            if conversation_id:
                self.db.log_tool_call(
                    conversation_id=conversation_id,
                    tool_name=tool.name,
                    args=args,
                    status=status,
                    result=result if status == "ok" else None,
                    error=error,
                    started_at_ms=started,
                    duration_ms=duration,
                )


def build_default_tool_registry(config: ButlerConfig, db: ButlerDB) -> ToolRegistry:
    sandbox = PathSandbox.from_strings(config.allowed_roots)
    tools: dict[str, Tool] = {}

    def add(tool: Tool) -> None:
        tools[tool.name] = tool

    for t in system_tools.build():
        add(t)
    for t in files_tools.build():
        add(t)
    for t in notes_tools.build():
        add(t)
    for t in index_tools.build():
        add(t)
    for t in weather_tools.build():
        add(t)
    for t in web_tools.build():
        add(t)
    for t in os_tools.build():
        add(t)
    for t in spotify_tools.build():
        add(t)

    return ToolRegistry(config=config, db=db, sandbox=sandbox, tools=tools)
