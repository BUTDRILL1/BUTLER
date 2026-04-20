from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from butler.agent.parsing import ParseError, parse_action_outcome_with_normalization
from butler.agent.prompting import (
    build_chat_system_prompt,
    build_repair_prompt_format,
    build_system_prompt,
)
from butler.agent.provider import GeminiProvider, OllamaProvider
from butler.agent.schema import ClarifyAction, FinalAction, ToolCallAction
from butler.config import ButlerConfig
from butler.db import ButlerDB
from butler.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_FACTUAL_CLEAN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(?:can you|could you|would you|will you|please)\s+", re.IGNORECASE),
    re.compile(r"^(?:show me|tell me|read me|give me|find me|look up|search for|check)\s+", re.IGNORECASE),
    re.compile(r"^(?:what is|what's|whats|what are|who is|who's)\s+", re.IGNORECASE),
    re.compile(r"^(?:top\s+\d+\s+)?(?:latest|current|today's|today|recent)\s+", re.IGNORECASE),
]
_RE_HEADLINE = re.compile(r"\bheadline\b", re.IGNORECASE)
_RE_SPACES = re.compile(r"\s+")


class AgentError(Exception):
    pass


_REFUSAL_PHRASES = (
    "unable to access external tools",
    "i can't browse",
    "i cannot browse",
    "i don't have access",
    "i do not have access",
    "external tools or information",
)


def _looks_like_plain_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("{") or stripped.startswith("[") or stripped.startswith("```"):
        return False
    return True


def _looks_like_refusal(text: str) -> bool:
    t = text.strip().lower()
    return any(p in t for p in _REFUSAL_PHRASES)


def _log_parse_event(
    *,
    model: str,
    parse_stage: str,
    parse_confidence: str,
    validation_error_count: int,
    repair_attempts: int,
) -> None:
    logger.debug(
        "parse_event model=%s parse_stage=%s parse_confidence=%s validation_error_count=%d repair_attempts=%d",
        model,
        parse_stage,
        parse_confidence,
        validation_error_count,
        repair_attempts,
    )


def _has_entity(text: str) -> bool:
    words = text.split()
    # If any word is capitalized (excluding the first word or I)
    if any(word.istitle() for word in words[1:] if word.lower() != 'i'):
        return True
    if any(char.isdigit() for char in text):
        return True
    # Specific known software/tech entities that users typically type in lowercase
    tech_keywords = ['xsa', 'bert', 'roberta', 'deberta', 'llama', 'gpt', 'iphone', 'macbook', 'mba', 'bits']
    if any(t in text.lower().split() for t in tech_keywords):
        return True
    return False


def _needs_external_info(text: str) -> bool:
    t = text.lower()
    strong = [
        "latest",
        "news",
        "current",
        "today",
        "today's",
        "yesterday",
        "recent",
        "price",
        "release",
        "update",
        "speech",
        "headline",
        "headlines",
        "breaking",
    ]
    weak = [
        "who is",
        "who was",
        "what is",
        "what are",
        "when was",
        "where is",
        "compare",
        "difference",
        "know about",
        "tell me about",
        "explain",
    ]

    if any(w in t for w in strong):
        return True
    if any(w in t for w in weak) and _has_entity(text):
        return True
    return False


def _select_tool(query: str) -> str:
    q = query.lower()
    if any(x in q for x in ["file", "document", "notes"]):
        return "files.search"
    if any(x in q for x in ["latest", "news", "price", "release", "update"]):
        return "web.search"
    return "web.search"


def _looks_like_casual_chat(text: str) -> bool:
    t = text.lower().strip()
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "bye", "thanks", "thank you"]
    if any(t == g or t.startswith(g + " ") or t.startswith(g + ",") for g in greetings):
        if "?" not in t and not _needs_external_info(text):
            return True

    casual_phrases = (
        "do you know my name",
        "who am i",
        "how are you",
        "what's up",
        "whats up",
        "tell me about yourself",
        "what can you do",
        "are you there",
        "can we chat",
        "how's it going",
        "hows it going",
        "remember my name",
    )
    if any(phrase in t for phrase in casual_phrases):
        return True

    if t.startswith("say ") and any(word in t for word in ("hi", "hello", "hey", "thanks", "thank you")):
        return True
    return False


def _looks_like_action_request(text: str) -> bool:
    t = text.lower()
    return any(
        word in t
        for word in (
            "search",
            "find",
            "index",
            "note",
            "notes",
            "file",
            "files",
            "read",
            "list",
            "roots",
            "config",
            "summarize",
            "summary",
            "recap",
            "details",
        )
    )


def _looks_like_weather_request(text: str) -> bool:
    t = text.lower()
    weather_terms = ("weather", "forecast", "temperature", "rain", "raining", "sunny", "humidity", "wind", "snow", "storm", "precipitation")
    return any(term in t for term in weather_terms)


def _classify_turn(text: str) -> tuple[str, str]:
    if _looks_like_casual_chat(text):
        return "CHAT", "casual_chat"
    if _looks_like_weather_request(text):
        return "ACTION", "weather_request"
    if _needs_external_info(text):
        return "FACTUAL_SEARCH", "current_or_recent_info"
    if _looks_like_action_request(text):
        return "ACTION", "tool_request"
    return "ACTION", "default_action"


def _clean_factual_search_query(text: str) -> str:
    cleaned = text.strip()
    changed = True
    while changed:
        changed = False
        for pattern in _FACTUAL_CLEAN_PATTERNS:
            new_cleaned = pattern.sub("", cleaned)
            if new_cleaned != cleaned:
                cleaned = new_cleaned.strip()
                changed = True

    cleaned = _RE_HEADLINE.sub("headlines", cleaned)
    cleaned = _RE_SPACES.sub(" ", cleaned).strip(" .,:;?-")
    return cleaned or text.strip()


@dataclass
class AgentRuntime:
    config: ButlerConfig
    db: ButlerDB
    tools: ToolRegistry
    provider: Any = None
    conversation_id: str | None = None
    confirm_tool: Callable[[str, dict[str, Any]], bool] | None = None
    _action_system_prompt: str = field(init=False, repr=False, compare=False, default="")
    _chat_system_prompt: str = field(init=False, repr=False, compare=False, default="")
    _has_chat_stream: bool = field(init=False, repr=False, compare=False, default=False)

    def __post_init__(self) -> None:
        if self.provider is None:
            if self.config.provider == "gemini":
                self.provider = GeminiProvider(
                    api_key=self.config.gemini_api_key,
                    model=self.config.model,
                    timeout_seconds=self.config.model_timeout_seconds,
                    retry_count=self.config.model_retry_count,
                    total_timeout_seconds=self.config.model_total_timeout_seconds,
                )
            else:
                self.provider = OllamaProvider(
                    base_url=self.config.ollama_url,
                    model=self.config.model,
                    timeout_seconds=self.config.model_timeout_seconds,
                    retry_count=self.config.model_retry_count,
                    total_timeout_seconds=self.config.model_total_timeout_seconds,
                )
        self._action_system_prompt = build_system_prompt(
            assistant_name=self.config.assistant_name,
            tools=self.tools.describe(),
        )
        self._chat_system_prompt = build_chat_system_prompt(assistant_name=self.config.assistant_name)
        self._has_chat_stream = hasattr(self.provider, "chat_stream")
        if self.conversation_id is None:
            existing = self.db.get_last_conversation(max_age_hours=24)
            self.conversation_id = existing or self.db.new_conversation()

    def _chat_history_messages(self, limit: int = 3) -> list[dict[str, Any]]:
        assert self.conversation_id is not None
        history = self.db.list_messages(self.conversation_id, limit=max(1, limit * 2))
        filtered: list[dict[str, Any]] = []
        for message in history:
            role = message.get("role", "")
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            stripped = content.strip()
            if stripped.startswith("{") or stripped.startswith("TOOL_"):
                continue
            if role == "assistant" and _looks_like_refusal(content):
                continue
            filtered.append(message)
        return filtered[-limit:]

    def _provider_chat(self, messages: list[dict[str, Any]], *, temperature: float, model: str) -> str:
        return self.provider.chat(messages, temperature=temperature, model=model)

    def _provider_chat_stream(self, messages: list[dict[str, Any]], *, temperature: float, model: str) -> Any:
        if self._has_chat_stream:
            yield from self.provider.chat_stream(messages, temperature=temperature, model=model)
            return
        yield self._provider_chat(messages, temperature=temperature, model=model)

    def _chat_mode_reply(self, model: str, user_text: str) -> str:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self._chat_system_prompt}]
        messages.extend(self._chat_history_messages())
        try:
            return self._provider_chat(messages, temperature=0.2, model=model)
        except Exception as e:  # noqa: BLE001
            logger.warning("chat_mode_failed model=%s error=%s", model, e)
            return "Sorry, I had trouble understanding that. Try rephrasing."

    def _chat_mode_reply_stream(self, model: str) -> Any:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self._chat_system_prompt}]
        messages.extend(self._chat_history_messages())
        try:
            started = time.perf_counter()
            first_token = True
            for token in self._provider_chat_stream(messages, temperature=0.2, model=model):
                if first_token:
                    logger.info(
                        "chat_first_token conversation_id=%s model=%s latency_ms=%d",
                        self.conversation_id,
                        model,
                        int((time.perf_counter() - started) * 1000),
                    )
                    first_token = False
                yield token
        except Exception as e:
            logger.warning("chat_stream_failed model=%s error=%s", model, e)
            yield "Sorry, I had trouble understanding that. Try rephrasing."

    def _stream_summarize(self, user_text: str, results_json: dict) -> Any:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self._chat_system_prompt}]
        messages.append({
            "role": "user",
            "content": (
                f"Boss asked: \"{user_text}\"\n\n"
                "Here are the search results. Reply directly to Boss in your normal BUTLER persona. Start your response immediately with the answer. Do not use preambles like 'I understand' or 'Here are the results'. Extract the direct answer, not a source list.\n"
                "If the results show weather, give the current conditions directly.\n"
                "If the results show news, give the most relevant answer or the top 3 headlines.\n"
                "Only use information from these results. Mention sources briefly only when helpful.\n\n"
                f"RESULTS:\n{json.dumps(results_json, ensure_ascii=False)}"
            )
        })
        try:
            started = time.perf_counter()
            first_token = True
            for token in self._provider_chat_stream(messages, temperature=0.2, model=self.config.chat_model):
                if first_token:
                    logger.info(
                        "factual_search_first_token conversation_id=%s model=%s latency_ms=%d",
                        self.conversation_id,
                        self.config.chat_model,
                        int((time.perf_counter() - started) * 1000),
                    )
                    first_token = False
                yield token
        except Exception as e:
            logger.warning("stream_summarize_failed error=%s", e)
            yield self._chat_mode_reply(self.config.chat_model, user_text)

    def _log_turn_route(self, user_text: str, route: str, reason: str) -> None:
        logger.info(
            "turn_route conversation_id=%s route=%s reason=%s chars=%d",
            self.conversation_id,
            route,
            reason,
            len(user_text.strip()),
        )

    def chat_once(self, user_text: str) -> str:
        return "".join(list(self.chat_once_stream(user_text)))

    def chat_once_stream(self, user_text: str) -> Any:
        try:
            yield from self._chat_once_stream_impl(user_text)
        finally:
            self.db.flush_turn()

    def _chat_once_stream_impl(self, user_text: str) -> Any:
        assert self.conversation_id is not None
        turn_started = time.perf_counter()
        self.db.add_message(self.conversation_id, "user", user_text)

        route, reason = _classify_turn(user_text)
        self._log_turn_route(user_text, route, reason)

        if route == "CHAT":
            full_response = ""
            for token in self._chat_mode_reply_stream(self.config.chat_model):
                full_response += token
                yield token
            self.db.add_message(self.conversation_id, "assistant", full_response)
            logger.info(
                "turn_complete conversation_id=%s route=%s duration_ms=%d outcome=chat",
                self.conversation_id,
                route,
                int((time.perf_counter() - turn_started) * 1000),
            )
            return

        # Pre-Emptive System Execution Variables
        force_system_tool = False
        tool_to_call = ""
        search_query = user_text
        if route == "FACTUAL_SEARCH":
            force_system_tool = True
            tool_to_call = _select_tool(user_text)
            search_query = _clean_factual_search_query(user_text)

        # Action Mode: attempt tool use with strict JSON action contract.
        tool_iterations = 0
        web_calls_this_turn = 0
        did_summarize_files_search = False
        awaiting_files_search_summary = False
        last_files_search_result: dict[str, Any] | None = None
        
        did_summarize_web_search = False
        awaiting_web_search_summary = False
        last_web_search_result: dict[str, Any] | None = None

        def fallback_files_search_summary() -> str:
            payload = last_files_search_result or {}
            results = payload.get("results") or []
            if not results:
                return "No matches found. Try different keywords or run /index sync."

            top = results[:3]
            best = top[0]
            best_line = f"{best.get('path', '')} - {best.get('snippet', '')}".strip()

            lines: list[str] = ["Best match:", best_line]
            if len(top) > 1:
                lines.extend(["", "Other matches:"])
                for idx, item in enumerate(top[1:3], start=1):
                    line = f"{item.get('path', '')} - {item.get('snippet', '')}".strip()
                    lines.append(f"{idx}. {line}")
            return "\n".join(lines).strip()
            
        def fallback_web_search_summary() -> str:
            payload = last_web_search_result or {}
            results = payload.get("results") or []
            if not results:
                return "No web search results found."
            lines: list[str] = ["Web Search Results:"]
            for idx, item in enumerate(results[:3], start=1):
                snippet = item.get("snippet") or item.get("url", "")
                line = f"{item.get('title', '')} - {snippet}".strip()
                lines.append(f"{idx}. {line}")
            return "\n".join(lines).strip()

        history = self.db.list_messages(self.conversation_id, limit=6)
        action_messages: list[dict[str, Any]] = [{"role": "system", "content": self._action_system_prompt}]
        action_messages.extend(history)
        if route == "ACTION" and reason == "weather_request":
            action_messages.append(
                {
                    "role": "user",
                    "content": (
                        "This is a weather request. Use weather.current.\n"
                        "Prefer the location from the user's message.\n"
                        "If none is provided, the tool may fall back to home_location or IP geolocation."
                    ),
                }
            )

        while tool_iterations <= self.config.max_tool_iterations:
            if force_system_tool and not did_summarize_web_search and tool_to_call in ("web.search", "web.news"):
                try:
                    result = self.tools.call(tool_to_call, {"query": search_query}, conversation_id=self.conversation_id)
                except Exception as e:
                    logger.warning("system_tool_call_failed tool=%s error=%s", tool_to_call, e)
                    result = {}

                if not result.get("results"):
                    assistant_text = "I couldn't fetch live news right now. Try again in a bit."
                    self.db.add_message(self.conversation_id, "assistant", assistant_text)
                    yield assistant_text
                    logger.info(
                        "turn_complete conversation_id=%s route=%s duration_ms=%d outcome=factual_search_fallback",
                        self.conversation_id,
                        route,
                        int((time.perf_counter() - turn_started) * 1000),
                    )
                    return

                # Stream the summary directly - no JSON, no parsing, no timeout
                full_response = ""
                for token in self._stream_summarize(user_text, result):
                    full_response += token
                    yield token
                self.db.add_message(self.conversation_id, "assistant", full_response)
                logger.info(
                    "turn_complete conversation_id=%s route=%s duration_ms=%d outcome=factual_search_summary",
                    self.conversation_id,
                    route,
                    int((time.perf_counter() - turn_started) * 1000),
                )
                return
                
            try:
                content = self._provider_chat(action_messages, temperature=0.0, model=self.config.model)
            except Exception as e:  # noqa: BLE001
                logger.warning("model_call_failed model=%s error=%s", self.config.model, e)
                if awaiting_files_search_summary and last_files_search_result is not None:
                    assistant_text = fallback_files_search_summary()
                    self.db.add_message(self.conversation_id, "assistant", assistant_text)
                    yield assistant_text
                    logger.info(
                        "turn_complete conversation_id=%s route=%s duration_ms=%d outcome=files_search_fallback",
                        self.conversation_id,
                        route,
                        int((time.perf_counter() - turn_started) * 1000),
                    )
                    return
                if awaiting_web_search_summary and last_web_search_result is not None:
                    assistant_text = fallback_web_search_summary()
                    self.db.add_message(self.conversation_id, "assistant", assistant_text)
                    yield assistant_text
                    logger.info(
                        "turn_complete conversation_id=%s route=%s duration_ms=%d outcome=web_search_fallback",
                        self.conversation_id,
                        route,
                        int((time.perf_counter() - turn_started) * 1000),
                    )
                    return
                assistant_text = self._chat_mode_reply(self.config.chat_model, user_text)
                self.db.add_message(self.conversation_id, "assistant", assistant_text)
                yield assistant_text
                logger.info(
                    "turn_complete conversation_id=%s route=%s duration_ms=%d outcome=chat_fallback",
                    self.conversation_id,
                    route,
                    int((time.perf_counter() - turn_started) * 1000),
                )
                return

            if awaiting_files_search_summary and last_files_search_result is not None:
                assistant_text = fallback_files_search_summary()
                self.db.add_message(self.conversation_id, "assistant", assistant_text)
                yield assistant_text
                logger.info(
                    "turn_complete conversation_id=%s route=%s duration_ms=%d outcome=files_search_fallback",
                    self.conversation_id,
                    route,
                    int((time.perf_counter() - turn_started) * 1000),
                )
                return
            if awaiting_web_search_summary and last_web_search_result is not None:
                assistant_text = fallback_web_search_summary()
                self.db.add_message(self.conversation_id, "assistant", assistant_text)
                yield assistant_text
                logger.info(
                    "turn_complete conversation_id=%s route=%s duration_ms=%d outcome=web_search_fallback",
                    self.conversation_id,
                    route,
                    int((time.perf_counter() - turn_started) * 1000),
                )
                return

            if _looks_like_refusal(content):
                assistant_text = self._chat_mode_reply(self.config.chat_model, user_text)
                self.db.add_message(self.conversation_id, "assistant", assistant_text)
                yield assistant_text
                logger.info(
                    "turn_complete conversation_id=%s route=%s duration_ms=%d outcome=refusal_chat_fallback",
                    self.conversation_id,
                    route,
                    int((time.perf_counter() - turn_started) * 1000),
                )
                return

            if _looks_like_plain_text(content):
                # If there are absolutely no JSON markers in the output, don't waste 2 minutes
                # trying to run multiple LLM repair tasks to force it to JSONify a casual chat message.
                if "{" not in content and "[" not in content and "```" not in content:
                    logger.debug("Intercepted pure conversational text. Bypassing JSON repair.")
                    self.db.add_message(self.conversation_id, "assistant", content)
                    yield content
                    return

            repair_attempts = 0
            validation_error_count = 0
            parse_stage = "fallback"
            parse_confidence = "fallback"
            plain_text_candidate = content.strip() if _looks_like_plain_text(content) else None

            try:
                outcome = parse_action_outcome_with_normalization(content)
                action = outcome.action
                parse_stage = outcome.parse_stage
                parse_confidence = outcome.parse_confidence
            except ParseError as initial_error:
                validation_error_count = max(validation_error_count, initial_error.validation_error_count)

                repair_attempts = 1
                try:
                    repair_a = self._provider_chat(
                        [
                            {"role": "system", "content": "Return one valid JSON object only."},
                            {"role": "user", "content": build_repair_prompt_format(content)},
                        ],
                        temperature=0.0,
                        model=self.config.model,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("model_repair_a_failed model=%s error=%s", self.config.model, e)
                    if awaiting_files_search_summary and last_files_search_result is not None:
                        assistant_text = fallback_files_search_summary()
                        self.db.add_message(self.conversation_id, "assistant", assistant_text)
                        yield assistant_text
                        logger.info(
                            "turn_complete conversation_id=%s route=%s duration_ms=%d outcome=files_search_fallback",
                            self.conversation_id,
                            route,
                            int((time.perf_counter() - turn_started) * 1000),
                        )
                        return
                    if awaiting_web_search_summary and last_web_search_result is not None:
                        assistant_text = fallback_web_search_summary()
                        self.db.add_message(self.conversation_id, "assistant", assistant_text)
                        yield assistant_text
                        logger.info(
                            "turn_complete conversation_id=%s route=%s duration_ms=%d outcome=web_search_fallback",
                            self.conversation_id,
                            route,
                            int((time.perf_counter() - turn_started) * 1000),
                        )
                        return
                    assistant_text = self._chat_mode_reply(self.config.chat_model, user_text)
                    self.db.add_message(self.conversation_id, "assistant", assistant_text)
                    yield assistant_text
                    logger.info(
                        "turn_complete conversation_id=%s route=%s duration_ms=%d outcome=chat_fallback",
                        self.conversation_id,
                        route,
                        int((time.perf_counter() - turn_started) * 1000),
                    )
                    return

                try:
                    outcome = parse_action_outcome_with_normalization(repair_a)
                    action = outcome.action
                    parse_stage = "repair_a"
                    parse_confidence = "repaired"
                except ParseError as repair_a_error:
                    validation_error_count = max(validation_error_count, repair_a_error.validation_error_count)

                    _log_parse_event(
                        model=self.config.model,
                        parse_stage="fallback",
                        parse_confidence="fallback",
                        validation_error_count=validation_error_count,
                        repair_attempts=repair_attempts,
                    )
                    if awaiting_files_search_summary and last_files_search_result is not None:
                        assistant_text = fallback_files_search_summary()
                        self.db.add_message(self.conversation_id, "assistant", assistant_text)
                        yield assistant_text
                        logger.info(
                            "turn_complete conversation_id=%s route=%s duration_ms=%d outcome=files_search_fallback",
                            self.conversation_id,
                            route,
                            int((time.perf_counter() - turn_started) * 1000),
                        )
                        return
                    if awaiting_web_search_summary and last_web_search_result is not None:
                        assistant_text = fallback_web_search_summary()
                        self.db.add_message(self.conversation_id, "assistant", assistant_text)
                        yield assistant_text
                        logger.info(
                            "turn_complete conversation_id=%s route=%s duration_ms=%d outcome=web_search_fallback",
                            self.conversation_id,
                            route,
                            int((time.perf_counter() - turn_started) * 1000),
                        )
                        return
                    assistant_text = self._chat_mode_reply(self.config.chat_model, user_text)
                    self.db.add_message(self.conversation_id, "assistant", assistant_text)
                    yield assistant_text
                    logger.info(
                        "turn_complete conversation_id=%s route=%s duration_ms=%d outcome=chat_fallback",
                        self.conversation_id,
                        route,
                        int((time.perf_counter() - turn_started) * 1000),
                    )
                    return

            _log_parse_event(
                model=self.config.model,
                parse_stage=parse_stage,
                parse_confidence=parse_confidence,
                validation_error_count=validation_error_count,
                repair_attempts=repair_attempts,
            )

            if isinstance(action, FinalAction):
                assistant_text = action.content
                self.db.add_message(self.conversation_id, "assistant", assistant_text)
                yield assistant_text
                return

            if isinstance(action, ClarifyAction):
                if awaiting_files_search_summary and last_files_search_result is not None:
                    assistant_text = fallback_files_search_summary()
                    self.db.add_message(self.conversation_id, "assistant", assistant_text)
                    yield assistant_text
                    return
                if awaiting_web_search_summary and last_web_search_result is not None:
                    assistant_text = fallback_web_search_summary()
                    self.db.add_message(self.conversation_id, "assistant", assistant_text)
                    yield assistant_text
                    return
                # If the user didn't ask for tool-like work, treat "clarify what tools to use"
                # responses as an Action Mode failure and fall back to Chat Mode.
                if tool_iterations == 0:
                    assistant_text = self._chat_mode_reply(self.config.chat_model, user_text)
                    self.db.add_message(self.conversation_id, "assistant", assistant_text)
                    yield assistant_text
                    return
                if action.choices:
                    choices = " / ".join(action.choices[:6])
                    assistant_text = f"{action.question} ({choices})"
                    self.db.add_message(self.conversation_id, "assistant", assistant_text)
                    yield assistant_text
                    return
                assistant_text = action.question
                self.db.add_message(self.conversation_id, "assistant", assistant_text)
                yield assistant_text
                return

            if isinstance(action, ToolCallAction):
                if awaiting_files_search_summary and last_files_search_result is not None:
                    assistant_text = fallback_files_search_summary()
                    self.db.add_message(self.conversation_id, "assistant", assistant_text)
                    yield assistant_text
                    logger.info(
                        "turn_complete conversation_id=%s route=%s duration_ms=%d outcome=files_search_fallback",
                        self.conversation_id,
                        route,
                        int((time.perf_counter() - turn_started) * 1000),
                    )
                    return
                if awaiting_web_search_summary and last_web_search_result is not None:
                    assistant_text = fallback_web_search_summary()
                    self.db.add_message(self.conversation_id, "assistant", assistant_text)
                    yield assistant_text
                    logger.info(
                        "turn_complete conversation_id=%s route=%s duration_ms=%d outcome=web_search_fallback",
                        self.conversation_id,
                        route,
                        int((time.perf_counter() - turn_started) * 1000),
                    )
                    return

                tool_iterations += 1
                tool = self.tools.tools.get(action.name)
                if tool is None:
                    assistant_text = self._chat_mode_reply(self.config.chat_model, user_text)
                    self.db.add_message(self.conversation_id, "assistant", assistant_text)
                    yield assistant_text
                    logger.info(
                        "turn_complete conversation_id=%s route=%s duration_ms=%d outcome=chat_fallback",
                        self.conversation_id,
                        route,
                        int((time.perf_counter() - turn_started) * 1000),
                    )
                    return
                    
                if action.name in ("web.search", "web.read"):
                    web_calls_this_turn += 1
                    if web_calls_this_turn > 2:
                        action_messages.append({"role": "assistant", "content": json.dumps(action.model_dump(), ensure_ascii=False)})
                        action_messages.append({"role": "user", "content": f"TOOL_ERROR {action.name}: Hard limit of 2 web calls per turn reached to prevent scraping loops."})
                        continue

                if tool.side_effect and self.config.confirm_writes:
                    allowed = False
                    if self.confirm_tool is not None:
                        try:
                            allowed = bool(self.confirm_tool(action.name, action.arguments))
                        except Exception:
                            allowed = False
                    if not allowed:
                        action_messages.append({"role": "assistant", "content": json.dumps(action.model_dump(), ensure_ascii=False)})
                        action_messages.append({"role": "user", "content": f"TOOL_DENIED {action.name}: user did not approve this action"})
                        continue

                try:
                    result = self.tools.call(action.name, action.arguments, conversation_id=self.conversation_id)
                except Exception as e:  # noqa: BLE001
                    action_messages.append({"role": "assistant", "content": json.dumps(action.model_dump(), ensure_ascii=False)})
                    action_messages.append({"role": "user", "content": f"TOOL_ERROR {action.name}: {e}"})
                    continue

                action_messages.append({"role": "assistant", "content": json.dumps(action.model_dump(), ensure_ascii=False)})
                action_messages.append({"role": "user", "content": f"TOOL_RESULT {action.name}: {json.dumps(result, ensure_ascii=False)}"})

                if action.name == "files.search":
                    last_files_search_result = result
                    if not did_summarize_files_search:
                        action_messages.append(
                            {
                                "role": "user",
                                "content": "Summarize these search results clearly for the user:\n"
                                "- Highlight best match\n"
                                "- If multiple results: list top 3\n"
                                "- Keep it short\n"
                                "Return a FINAL action JSON only; do not call tools.\n\n"
                                f"RESULTS_JSON: {json.dumps(result, ensure_ascii=False)}",
                            }
                        )
                        did_summarize_files_search = True
                        awaiting_files_search_summary = True
                        
                if action.name == "web.search":
                    last_web_search_result = result
                    if not did_summarize_web_search:
                        action_messages.append(
                            {
                                "role": "user",
                                "content": "Summarize these web search results cleanly for the user:\n"
                                "- Write a brief coherent answer based on the results.\n"
                                "- Include source citations.\n"
                                "Return a FINAL action JSON only; do not call tools.\n\n"
                                f"RESULTS_JSON: {json.dumps(result, ensure_ascii=False)}",
                            }
                        )
                        did_summarize_web_search = True
                        awaiting_web_search_summary = True
                continue

        if awaiting_files_search_summary and last_files_search_result is not None:
            assistant_text = fallback_files_search_summary()
            self.db.add_message(self.conversation_id, "assistant", assistant_text)
            yield assistant_text
            logger.info(
                "turn_complete conversation_id=%s route=%s duration_ms=%d outcome=files_search_fallback",
                self.conversation_id,
                route,
                int((time.perf_counter() - turn_started) * 1000),
            )
            return
        if awaiting_web_search_summary and last_web_search_result is not None:
            assistant_text = fallback_web_search_summary()
            self.db.add_message(self.conversation_id, "assistant", assistant_text)
            yield assistant_text
            logger.info(
                "turn_complete conversation_id=%s route=%s duration_ms=%d outcome=web_search_fallback",
                self.conversation_id,
                route,
                int((time.perf_counter() - turn_started) * 1000),
            )
            return

        _log_parse_event(
            model=self.config.model,
            parse_stage="fallback",
            parse_confidence="fallback",
            validation_error_count=0,
            repair_attempts=0,
        )
        assistant_text = self._chat_mode_reply(self.config.chat_model, user_text)
        self.db.add_message(self.conversation_id, "assistant", assistant_text)
        yield assistant_text
        logger.info(
            "turn_complete conversation_id=%s route=%s duration_ms=%d outcome=chat_fallback",
            self.conversation_id,
            route,
            int((time.perf_counter() - turn_started) * 1000),
        )
        return
