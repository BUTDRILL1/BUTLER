from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from butler.agent.loop import AgentRuntime
from butler.config import ButlerConfig
from butler.db import ButlerDB
from butler.tools.registry import build_default_tool_registry


@dataclass
class InspectingProvider:
    responses: list[str]
    seen_system_prompts: list[str]
    seen_models: list[str] = field(default_factory=list)

    def chat(self, messages: list[dict[str, Any]], *, temperature: float = 0.2, model: str | None = None) -> str:
        sys_msg = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
        if isinstance(sys_msg, str):
            self.seen_system_prompts.append(sys_msg)
        if isinstance(model, str):
            self.seen_models.append(model)
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
        seen_models=[],
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
            '{"type":"tool_call"}',
            "still not json",
            "Hi!",
        ],
        seen_system_prompts=seen,
        seen_models=[],
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
        seen_models=[],
    )
    runtime = _runtime(tmp_path, monkeypatch, provider)
    out = runtime.chat_once("search hi")
    assert out == "Hey! Hi there."


def test_action_mode_unknown_tool_for_smalltalk_falls_back_to_chat_mode(tmp_path, monkeypatch) -> None:
    seen: list[str] = []
    provider = InspectingProvider(
        responses=[
            '{"type":"tool_call","name":"not_a_real_tool","arguments":{}}',
            '{"type":"final","content":"Hi!"}',
        ],
        seen_system_prompts=seen,
        seen_models=[],
    )
    runtime = _runtime(tmp_path, monkeypatch, provider)
    out = runtime.chat_once("search hi")
    assert out == '{"type":"final","content":"Hi!"}'


def test_smalltalk_goes_directly_to_chat_mode(tmp_path, monkeypatch) -> None:
    seen: list[str] = []
    seen_models: list[str] = []
    provider = InspectingProvider(responses=["Hello there."], seen_system_prompts=seen, seen_models=seen_models)
    runtime = _runtime(tmp_path, monkeypatch, provider)
    out = runtime.chat_once("Bro do you know my name")
    assert out == "Hello there."
    assert any("Do not mention tools" in p for p in seen)
    assert seen_models == ["gemma:2b"]


def test_current_event_prompt_routes_to_web_search(tmp_path, monkeypatch) -> None:
    seen: list[str] = []
    provider = InspectingProvider(
        responses=["Hereâ€™s what I found in the news."],
        seen_system_prompts=seen,
        seen_models=[],
    )
    runtime = _runtime(tmp_path, monkeypatch, provider)

    def fake_call(tool_name: str, args: dict[str, Any], *, conversation_id: str | None = None):
        assert tool_name == "web.search"
        return {
            "query": args["query"],
            "results": [
                {"title": "Modi speech", "snippet": "Key points from the speech", "url": "https://example.com"}
            ],
            "cached": False,
        }

    runtime.tools.call = fake_call  # type: ignore[method-assign]

    out = runtime.chat_once("I heard that yesterday Modi ji gave a speech at around 8pm. what was it about?")
    assert out == "Hereâ€™s what I found in the news."


def test_summary_followup_routes_to_action_mode(tmp_path, monkeypatch) -> None:
    seen: list[str] = []
    seen_models: list[str] = []
    provider = InspectingProvider(
        responses=['{"type":"final","content":"Boss, here is the summary."}'],
        seen_system_prompts=seen,
        seen_models=seen_models,
    )
    runtime = _runtime(tmp_path, monkeypatch, provider)
    out = runtime.chat_once("yes summarize what all he said")
    assert out == "Boss, here is the summary."
    assert seen_models == ["mistral:7b-instruct"]
    assert any("You must respond with EXACTLY ONE JSON object" in p for p in seen)


def test_factual_summarizer_addresses_boss_directly(tmp_path, monkeypatch) -> None:
    from dataclasses import dataclass, field

    @dataclass
    class RecordingProvider:
        responses: list[str]
        seen_messages: list[list[dict[str, Any]]] = field(default_factory=list)
        seen_models: list[str] = field(default_factory=list)

        def chat(self, messages: list[dict[str, Any]], *, temperature: float = 0.2, model: str | None = None) -> str:
            self.seen_messages.append(messages)
            if isinstance(model, str):
                self.seen_models.append(model)
            return self.responses.pop(0)

    provider = RecordingProvider(responses=["Boss, here is the answer."])
    runtime = _runtime(tmp_path, monkeypatch, provider)  # type: ignore[arg-type]

    def fake_call(tool_name: str, args: dict[str, Any], *, conversation_id: str | None = None):
        assert tool_name == "web.search"
        return {
            "query": args["query"],
            "results": [
                {"title": "Speech summary", "snippet": "He discussed reforms and women empowerment.", "url": "https://example.com"}
            ],
            "cached": False,
        }

    runtime.tools.call = fake_call  # type: ignore[method-assign]

    out = runtime.chat_once("I heard Mr. Modi gave a speech on 18th april. find and summarize it to me")
    assert out == "Boss, here is the answer."
    assert provider.seen_models == ["gemma:2b"]
    assert any("Boss asked:" in m.get("content", "") for m in provider.seen_messages[0] if m.get("role") == "user")

