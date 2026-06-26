from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from butler.config import ButlerConfig
from butler.db import ButlerDB
from butler.sandbox import PathSandbox
from butler.tools.base import Tool, ToolCallRecord, ToolContext, ToolError, now_ms

if TYPE_CHECKING:
    from butler.agent.memory import MemoryStore
from butler.tools.impl import files as files_tools
from butler.tools.impl import notes as notes_tools
from butler.tools.impl import system as system_tools
from butler.tools.impl import weather as weather_tools
from butler.tools.impl import web as web_tools
from butler.tools.impl import spotify_control
from butler.tools.impl import os_control
from butler.tools.impl import reminders as reminders_tools
from butler.tools.impl import github as github_tools

logger = logging.getLogger(__name__)


import importlib.util
from pathlib import Path
from butler.paths import butler_home_dir

@dataclass
class ToolRegistry:
    config: ButlerConfig
    db: ButlerDB
    sandbox: PathSandbox
    memory: MemoryStore
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

    def reload(self) -> None:
        self.tools = _load_all_tools(self.config, self.db, self.memory)
        self.__post_init__()

    def describe(self) -> list[dict[str, Any]]:
        return self._cached_description

    def call(self, tool_name: str, args: dict[str, Any], *, conversation_id: str | None = None) -> dict[str, Any]:
        if tool_name not in self.tools:
            raise ToolError(f"Unknown tool: {tool_name}")
        tool = self.tools[tool_name]
        ctx = ToolContext(config=self.config, db=self.db, sandbox=self.sandbox, memory=self.memory, registry=self)

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


def _load_all_tools(config: ButlerConfig, db: ButlerDB, memory: MemoryStore) -> dict[str, Tool]:
    from butler.tools.impl import skills as skills_tools
    tools: dict[str, Tool] = {}

    def add(tool: Tool) -> None:
        tools[tool.name] = tool

    for t in system_tools.build(): add(t)
    for t in files_tools.build(): add(t)
    for t in notes_tools.build(): add(t)
    for t in weather_tools.build(): add(t)
    for t in web_tools.build(): add(t)
    for t in reminders_tools.build(): add(t)
    for t in spotify_control.build(): add(t)
    for t in os_control.build(): add(t)
    
    for t in github_tools.TOOLS: add(t)
    
    try:
        for t in skills_tools.build(): add(t)
    except Exception as e:
        logger.error("Failed to load skills_tools: %s", e)

    skills_dir = butler_home_dir() / "skills"
    if skills_dir.exists():
        for py_file in skills_dir.glob("*.py"):
            try:
                spec = importlib.util.spec_from_file_location(f"butler.skills.{py_file.stem}", py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, "build"):
                        for t in module.build():
                            add(t)
            except Exception as e:
                logger.error("Failed to load dynamic skill %s: %s", py_file, e)
                
    return tools


def build_default_tool_registry(config: ButlerConfig, db: ButlerDB, memory: MemoryStore) -> ToolRegistry:
    sandbox = PathSandbox.from_strings(config.allowed_roots)
    tools = _load_all_tools(config, db, memory)
    return ToolRegistry(config=config, db=db, sandbox=sandbox, memory=memory, tools=tools)
