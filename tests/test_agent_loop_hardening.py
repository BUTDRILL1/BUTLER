from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from butler.agent.loop import AgentRuntime
from butler.config import ButlerConfig
from butler.db import ButlerDB
from butler.tools.registry import build_default_tool_registry


@dataclass
class FakeProvider:
    responses: list[str]

    def chat(self, messages: list[dict[str, Any]], *, temperature: float = 0.2, model: str | None = None) -> str:
        if not self.responses:
            return '{"type":"final","content":"(no scripted response left)"}'
        return self.responses.pop(0)


class FailingProvider:
    def chat(self, messages: list[dict[str, Any]], *, temperature: float = 0.2, model: str | None = None) -> str:
        raise TimeoutError("simulated timeout")


def _runtime(tmp_path, monkeypatch, responses: list[str]) -> AgentRuntime:
    monkeypatch.setenv("BUTLER_HOME", str(tmp_path))
    cfg = ButlerConfig()
    db = ButlerDB.open(cfg)
    tools = build_default_tool_registry(cfg, db)
    return AgentRuntime(config=cfg, db=db, tools=tools, provider=FakeProvider(responses=responses))


def test_plain_text_first_then_failed_repairs_returns_plain_text(tmp_path, monkeypatch) -> None:
    runtime = _runtime(
        tmp_path,
        monkeypatch,
        responses=[
            "Hi! How can I assist you today?",
            '{"type":"tool_call"}',
            '{"type":"tool_call"}',
            "Hello from chat mode.",
        ],
    )
    out = runtime.chat_once("search hi")
    assert out == "Hi! How can I assist you today?"


def test_invalid_schema_all_attempts_returns_standard_fallback(tmp_path, monkeypatch) -> None:
    runtime = _runtime(
        tmp_path,
        monkeypatch,
        responses=[
            '{"type":"tool_call"}',
            '{"type":"tool_call"}',
            "Hi from chat mode.",
        ],
    )
    out = runtime.chat_once("search hi")
    assert out == "Hi from chat mode."


def test_repair_success_logs_repaired_confidence(tmp_path, monkeypatch, caplog) -> None:
    runtime = _runtime(
        tmp_path,
        monkeypatch,
        responses=[
            '{"type":"tool_call"}',
            '{"type":"final","content":"hello"}',
        ],
    )

    caplog.set_level(logging.DEBUG, logger="butler.agent.loop")
    out = runtime.chat_once("search hi")
    assert out == "hello"
    assert "parse_confidence=repaired" in caplog.text
    assert "parse_stage=repair_a" in caplog.text


def test_model_call_failure_returns_standard_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BUTLER_HOME", str(tmp_path))
    cfg = ButlerConfig()
    db = ButlerDB.open(cfg)
    tools = build_default_tool_registry(cfg, db)
    runtime = AgentRuntime(config=cfg, db=db, tools=tools, provider=FailingProvider())
    out = runtime.chat_once("search hi")
    assert out == "Sorry, I had trouble understanding that. Try rephrasing."


def test_smalltalk_fast_path_returns_chat_mode_first_response(tmp_path, monkeypatch) -> None:
    runtime = _runtime(
        tmp_path,
        monkeypatch,
        responses=[
            "Hi there!",
        ],
    )
    out = runtime.chat_once("say hi")
    assert out == "Hi there!"
