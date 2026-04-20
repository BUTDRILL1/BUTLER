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


def test_persona_keeps_boss_tone_and_no_tool_filler(tmp_path, monkeypatch) -> None:
    seen: list[str] = []
    seen_models: list[str] = []
    provider = InspectingProvider(responses=["Hello there."], seen_system_prompts=seen, seen_models=seen_models)
    runtime = _runtime(tmp_path, monkeypatch, provider)

    out = runtime.chat_once("Hi")

    assert out == "Hello there."
    assert any("The user is your Boss" in p for p in seen)
    assert any("Do not mention tools" in p for p in seen)
    assert all("datetime" not in p.lower() for p in seen)
    assert seen_models == ["gemma:2b"]


def test_factual_search_query_is_cleaned_before_web_search(tmp_path, monkeypatch) -> None:
    seen: list[str] = []
    provider = InspectingProvider(responses=["Top headlines found."], seen_system_prompts=seen, seen_models=[])
    runtime = _runtime(tmp_path, monkeypatch, provider)

    def fake_call(tool_name: str, args: dict[str, Any], *, conversation_id: str | None = None):
        assert tool_name == "web.search"
        assert args["query"].lower() == "top 3 news headlines"
        return {
            "query": args["query"],
            "results": [
                {"title": "Headline 1", "snippet": "Key point", "url": "https://example.com"}
            ],
            "cached": False,
        }

    runtime.tools.call = fake_call  # type: ignore[method-assign]

    out = runtime.chat_once("read me top 3 news headline")
    assert out == "Top headlines found."


def test_weather_prompt_routes_to_weather_tool(tmp_path, monkeypatch) -> None:
    seen: list[str] = []
    seen_models: list[str] = []
    provider = InspectingProvider(
        responses=[
            '{"type":"tool_call","name":"weather.current","arguments":{"location":"NOIDA sector 74, Uttar Pradesh, INDIA"}}',
            '{"type":"final","content":"It is 29°C and partly cloudy."}',
        ],
        seen_system_prompts=seen,
    )
    runtime = _runtime(tmp_path, monkeypatch, provider)

    def fake_call(tool_name: str, args: dict[str, Any], *, conversation_id: str | None = None):
        assert tool_name == "weather.current"
        assert args["location"] == "NOIDA sector 74, Uttar Pradesh, INDIA"
        return {
            "location": "NOIDA sector 74, Uttar Pradesh, INDIA",
            "resolved_from": "query",
            "current": {"temperature_c": 29, "condition": "Partly cloudy"},
        }

    runtime.tools.call = fake_call  # type: ignore[method-assign]

    out = runtime.chat_once("Can you check todays weather around me? i am in NOIDA sector 74, Uttar Pradesh, INDIA")
    assert out == "It is 29°C and partly cloudy."
