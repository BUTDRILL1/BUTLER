from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from butler.agent.loop import AgentRuntime
from butler.config import ButlerConfig
from butler.db import ButlerDB
from butler.tools.registry import build_default_tool_registry


@dataclass
class InspectingProvider:
    responses: list[str]
    seen_system_prompts: list[str]

    def chat(self, messages: list[dict[str, Any]], *, temperature: float = 0.2) -> str:
        sys_msg = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
        if isinstance(sys_msg, str):
            self.seen_system_prompts.append(sys_msg)
        if not self.responses:
            return "no more responses"
        return self.responses.pop(0)


def _runtime(tmp_path, monkeypatch, provider: InspectingProvider) -> AgentRuntime:
    monkeypatch.setenv("BUTLER_HOME", str(tmp_path))
    cfg = ButlerConfig()
    db = ButlerDB.open(cfg)
    tools = build_default_tool_registry(cfg, db)
    return AgentRuntime(config=cfg, db=db, tools=tools, provider=provider)


def test_refusal_in_action_mode_falls_back_to_chat_mode(tmp_path, monkeypatch) -> None:
    seen: list[str] = []
    provider = InspectingProvider(
        responses=[
            "I'm unable to access external tools or information, so I cannot help.",
            "Hello! Nice to meet you.",
        ],
        seen_system_prompts=seen,
    )
    runtime = _runtime(tmp_path, monkeypatch, provider)
    out = runtime.chat_once("search hi")
    assert out == "Hello! Nice to meet you."
    assert any("You must respond with EXACTLY ONE JSON object" in p for p in seen)
    assert any("Do not mention tools" in p for p in seen)


def test_invalid_action_schema_after_repairs_falls_back_to_chat_mode(tmp_path, monkeypatch) -> None:
    seen: list[str] = []
    provider = InspectingProvider(
        responses=[
            "not json",
            "still not json",
            '{"type":"tool_call"}',
            "Hi!",
        ],
        seen_system_prompts=seen,
    )
    runtime = _runtime(tmp_path, monkeypatch, provider)
    out = runtime.chat_once("search hi")
    assert out == "Hi!"


def test_action_mode_clarify_for_tool_input_returns_clarify(tmp_path, monkeypatch) -> None:
    seen: list[str] = []
    provider = InspectingProvider(
        responses=[
            '{"type":"clarify","question":"What would you like to know?","choices":["time","tools"]}',
            "Hey! Hi there.",
        ],
        seen_system_prompts=seen,
    )
    runtime = _runtime(tmp_path, monkeypatch, provider)
    out = runtime.chat_once("search hi")
    assert out == "What would you like to know? (time / tools)"


def test_action_mode_unknown_tool_for_smalltalk_falls_back_to_chat_mode(tmp_path, monkeypatch) -> None:
    seen: list[str] = []
    provider = InspectingProvider(
        responses=[
            '{"type":"tool_call","name":"not_a_real_tool","arguments":{}}',
            '{"type":"final","content":"Hi!"}',
        ],
        seen_system_prompts=seen,
    )
    runtime = _runtime(tmp_path, monkeypatch, provider)
    out = runtime.chat_once("search hi")
    assert out == "Hi!"


def test_smalltalk_goes_directly_to_chat_mode(tmp_path, monkeypatch) -> None:
    seen: list[str] = []
    provider = InspectingProvider(
        responses=["Hello there."],
        seen_system_prompts=seen,
    )
    runtime = _runtime(tmp_path, monkeypatch, provider)
    out = runtime.chat_once("Say hi")
    assert out == "Hello there."
    assert any("Do not mention tools" in p for p in seen)
