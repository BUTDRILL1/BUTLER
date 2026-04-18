from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from butler.agent.parsing import ParseError, parse_action_outcome_with_normalization
from butler.agent.prompting import (
    build_chat_system_prompt,
    build_repair_prompt_format,
    build_repair_prompt_schema,
    build_system_prompt,
    build_router_system_prompt,
)
from butler.agent.provider import OllamaProvider
from butler.agent.schema import ClarifyAction, FinalAction, ToolCallAction
from butler.config import ButlerConfig
from butler.db import ButlerDB
from butler.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


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


def _validation_hint(error: ParseError) -> str:
    if error.validation_error_count > 0:
        return f"Schema validation failed with {error.validation_error_count} missing/invalid fields."
    return error.message


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
    strong = ["latest", "news", "current", "today", "recent", "price", "release", "update"]
    weak = ["who is", "who was", "what is", "what are", "when was", "where is", "compare", "difference", "know about", "tell me about", "explain"]

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


def _fast_classify(text: str) -> str | None:
    t = text.lower().strip()
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "bye", "thanks", "thank you"]
    if any(t == g or t.startswith(g + " ") or t.startswith(g + ",") for g in greetings):
        if "?" not in t and not _needs_external_info(text):
            return "CHAT"
    if _needs_external_info(text):
        return "ACTION"
    return None


@dataclass
class AgentRuntime:
    config: ButlerConfig
    db: ButlerDB
    tools: ToolRegistry
    provider: OllamaProvider | None = None
    conversation_id: str | None = None
    confirm_tool: Callable[[str, dict[str, Any]], bool] | None = None

    def __post_init__(self) -> None:
        if self.provider is None:
            self.provider = OllamaProvider(
                base_url=self.config.ollama_url,
                timeout_seconds=self.config.model_timeout_seconds,
                retry_count=self.config.model_retry_count,
                total_timeout_seconds=self.config.model_total_timeout_seconds,
            )
        if self.conversation_id is None:
            existing = self.db.get_last_conversation(max_age_hours=24)
            self.conversation_id = existing or self.db.new_conversation()

    def _model_messages(self) -> list[dict[str, Any]]:
        assert self.conversation_id is not None
        system = build_system_prompt(assistant_name=self.config.assistant_name, tools=self.tools.describe())
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        history = self.db.list_messages(self.conversation_id)
        messages.extend(history[-10:])
        return messages

    def _chat_mode_reply(self, model: str, user_text: str) -> str:
        system = build_chat_system_prompt(assistant_name=self.config.assistant_name)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        history = self.db.list_messages(self.conversation_id)
        messages.extend(history[-10:])
        try:
            return self.provider.chat(model, messages, temperature=0.2)
        except Exception as e:  # noqa: BLE001
            logger.warning("chat_mode_failed model=%s error=%s", model, e)
            return "Apologies, Boss. I couldn't process that. Could you rephrase?"

    def _chat_mode_reply_stream(self, model: str) -> Any:
        system = build_chat_system_prompt(assistant_name=self.config.assistant_name)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        history = self.db.list_messages(self.conversation_id)
        messages.extend(history[-10:])
        try:
            yield from self.provider.chat_stream(model, messages, temperature=0.2)
        except Exception as e:
            logger.warning("chat_stream_failed model=%s error=%s", model, e)
            yield "Apologies, Boss. I couldn't process that. Could you rephrase?"

    def _stream_summarize(self, user_text: str, results_json: dict) -> Any:
        system = build_chat_system_prompt(assistant_name=self.config.assistant_name)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        history = self.db.list_messages(self.conversation_id)
        messages.extend(history[-6:])
        messages.append({
            "role": "user",
            "content": (
                f"The user asked: \"{user_text}\"\n\n"
                "Here are search results. Summarize them into a concise, helpful answer.\n"
                "Only use information from these results. Cite sources when relevant.\n\n"
                f"RESULTS:\n{json.dumps(results_json, ensure_ascii=False)}"
            )
        })
        try:
            yield from self.provider.chat_stream(
                self.config.smart_model, messages, temperature=0.2
            )
        except Exception as e:
            logger.warning("stream_summarize_failed error=%s", e)
            yield "No luck finding that one, Boss. Try rephrasing or being more specific."

    def _route_prompt(self, user_text: str) -> str:
        system = build_router_system_prompt()
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        history = self.db.list_messages(self.conversation_id)
        # Limit router context to last 4 messages to keep classification focused
        messages.extend(history[-4:])
        try:
            decision = self.provider.chat(self.config.fast_model, messages, temperature=0.0).strip().upper()
        except Exception as e:
            logger.warning("route_failed model=%s error=%s", self.config.fast_model, e)
            return "ACTION"
            
        if decision not in ("CHAT", "ACTION"):
            return "ACTION"
        return decision

    def chat_once(self, user_text: str) -> str:
        return "".join(list(self.chat_once_stream(user_text)))

    def chat_once_stream(self, user_text: str) -> Any:
        assert self.conversation_id is not None
        self.db.add_message(self.conversation_id, "user", user_text)

        route = _fast_classify(user_text)
        if route is None:
            route = self._route_prompt(user_text)

        if route == "CHAT":
            full_response = ""
            for token in self._chat_mode_reply_stream(self.config.smart_model):
                full_response += token
                yield token
            self.db.add_message(self.conversation_id, "assistant", full_response)
            return

        # Pre-Emptive System Execution Variables
        force_system_tool = False
        tool_to_call = ""
        if _needs_external_info(user_text):
            force_system_tool = True
            tool_to_call = _select_tool(user_text)

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
                line = f"{item.get('title', '')} - {item.get('url', '')}".strip()
                lines.append(f"{idx}. {line}")
            return "\n".join(lines).strip()

        while tool_iterations <= self.config.max_tool_iterations:
            system = build_system_prompt(assistant_name=self.config.assistant_name, tools=self.tools.describe())
            # Keep action mode per-turn and local, to avoid polluting future turns with repair/tool chatter.
            action_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
            history = self.db.list_messages(self.conversation_id)
            action_messages.extend(history[-10:])
            break

        # Action loop runs on a local message list.
        forced_tool_attempts = 0
        while tool_iterations <= self.config.max_tool_iterations:
            if force_system_tool and not did_summarize_web_search and tool_to_call == "web.search":
                forced_tool_attempts += 1
                if forced_tool_attempts > 1:
                    assistant_text = "No luck finding that one, Boss. Try rephrasing or being more specific."
                    self.db.add_message(self.conversation_id, "assistant", assistant_text)
                    yield assistant_text
                    return
                    
                try:
                    result = self.tools.call("web.search", {"query": user_text}, conversation_id=self.conversation_id)
                except Exception as e:
                    logger.warning("system_tool_call_failed tool=web.search error=%s", e)
                    result = {}
                
                if not result.get("results") or len(result.get("results", [])) < 2:
                    assistant_text = "No luck finding that one, Boss. Try rephrasing or being more specific."
                    self.db.add_message(self.conversation_id, "assistant", assistant_text)
                    yield assistant_text
                    return
                    
                # Stream the summary directly - no JSON, no parsing, no timeout
                full_response = ""
                for token in self._stream_summarize(user_text, result):
                    full_response += token
                    yield token
                self.db.add_message(self.conversation_id, "assistant", full_response)
                return
                
            try:
                content = self.provider.chat(self.config.smart_model, action_messages, temperature=0.0)
            except Exception as e:  # noqa: BLE001
                logger.warning("model_call_failed model=%s error=%s", self.config.smart_model, e)
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
                assistant_text = self._chat_mode_reply(self.config.smart_model, user_text)
                self.db.add_message(self.conversation_id, "assistant", assistant_text)
                yield assistant_text
                return

            if _looks_like_plain_text(content):
                # If there are absolutely no JSON markers in the output, don't waste 2 minutes
                # trying to run multiple LLM repair tasks to force it to JSONify a casual chat message.
                if "{" not in content and "[" not in content and "```" not in content:
                    logger.debug("Intercepted pure conversational text. Bypassing JSON repair.")
                    self.db.add_message(self.conversation_id, "assistant", content)
                    yield content
                    return
                
                if _looks_like_refusal(content):
                    assistant_text = self._chat_mode_reply(self.config.smart_model, user_text)
                    self.db.add_message(self.conversation_id, "assistant", assistant_text)
                    yield assistant_text
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
                    repair_a = self.provider.chat(
                        self.config.smart_model,
                        [
                            {"role": "system", "content": "Return one valid JSON object only."},
                            {"role": "user", "content": build_repair_prompt_format(content)},
                        ],
                        temperature=0.0,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("model_repair_a_failed model=%s error=%s", self.config.smart_model, e)
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
                    assistant_text = self._chat_mode_reply(self.config.smart_model, user_text)
                    self.db.add_message(self.conversation_id, "assistant", assistant_text)
                    yield assistant_text
                    return

                try:
                    outcome = parse_action_outcome_with_normalization(repair_a)
                    action = outcome.action
                    parse_stage = "repair_a"
                    parse_confidence = "repaired"
                except ParseError as repair_a_error:
                    validation_error_count = max(validation_error_count, repair_a_error.validation_error_count)

                    repair_attempts = 2
                    try:
                        repair_b = self.provider.chat(
                        self.config.smart_model,
                            [
                                {"role": "system", "content": "Return one valid JSON object only."},
                                {
                                    "role": "user",
                                    "content": build_repair_prompt_schema(
                                        repair_a,
                                        validation_hint=_validation_hint(repair_a_error),
                                    ),
                                },
                            ],
                            temperature=0.0,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning("model_repair_b_failed model=%s error=%s", self.config.smart_model, e)
                        _log_parse_event(
                            model=self.config.smart_model,
                            parse_stage="fallback",
                            parse_confidence="fallback",
                            validation_error_count=validation_error_count,
                            repair_attempts=repair_attempts,
                        )
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
                        assistant_text = self._chat_mode_reply(self.config.smart_model, user_text)
                        self.db.add_message(self.conversation_id, "assistant", assistant_text)
                        yield assistant_text
                        return

                    try:
                        outcome = parse_action_outcome_with_normalization(repair_b)
                        action = outcome.action
                        parse_stage = "repair_b"
                        parse_confidence = "repaired"
                    except ParseError as repair_b_error:
                        validation_error_count = max(
                            validation_error_count,
                            repair_b_error.validation_error_count,
                        )
                        _log_parse_event(
                            model=self.config.smart_model,
                            parse_stage="fallback",
                            parse_confidence="fallback",
                            validation_error_count=validation_error_count,
                            repair_attempts=repair_attempts,
                        )
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
                        assistant_text = self._chat_mode_reply(self.config.smart_model, user_text)
                        self.db.add_message(self.conversation_id, "assistant", assistant_text)
                        yield assistant_text
                        return

            _log_parse_event(
                model=self.config.smart_model,
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
                    assistant_text = self._chat_mode_reply(self.config.smart_model, user_text)
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
                    return
                if awaiting_web_search_summary and last_web_search_result is not None:
                    assistant_text = fallback_web_search_summary()
                    self.db.add_message(self.conversation_id, "assistant", assistant_text)
                    yield assistant_text
                    return

                tool_iterations += 1
                tool = self.tools.tools.get(action.name)
                if tool is None:
                    assistant_text = self._chat_mode_reply(self.config.smart_model, user_text)
                    self.db.add_message(self.conversation_id, "assistant", assistant_text)
                    yield assistant_text
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
            return
        if awaiting_web_search_summary and last_web_search_result is not None:
            assistant_text = fallback_web_search_summary()
            self.db.add_message(self.conversation_id, "assistant", assistant_text)
            yield assistant_text
            return

        _log_parse_event(
            model=self.config.smart_model,
            parse_stage="fallback",
            parse_confidence="fallback",
            validation_error_count=0,
            repair_attempts=0,
        )
        assistant_text = self._chat_mode_reply(self.config.smart_model, user_text)
        self.db.add_message(self.conversation_id, "assistant", assistant_text)
        yield assistant_text
        return
