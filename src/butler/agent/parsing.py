from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import TypeAdapter, ValidationError

from butler.agent.schema import Action

logger = logging.getLogger(__name__)

_ACTION_ADAPTER = TypeAdapter(Action)


ParseStage = Literal["direct_json", "json_string", "escaped_json"]


@dataclass(frozen=True)
class ParseOutcome:
    action: Action
    parse_stage: ParseStage
    parse_confidence: Literal["direct"]
    validation_error_count: int


@dataclass(frozen=True)
class ParseError(Exception):
    message: str
    raw_output: str
    cleaned_output: str
    validation_error_count: int = 0

    def __str__(self) -> str:  # pragma: no cover
        return self.message


_FENCE_LINE_RE = re.compile(r"(?m)^\s*```[a-zA-Z0-9_-]*\s*$")


def _strip_code_fences(text: str) -> str:
    # Remove ONLY fence delimiter lines. Keep the interior content.
    return _FENCE_LINE_RE.sub("", text).strip()


def _normalize_aliases(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)

    if "type" not in normalized and isinstance(normalized.get("action"), str):
        normalized["type"] = normalized["action"]

    if normalized.get("type") == "tool_call":
        if "arguments" not in normalized and isinstance(normalized.get("args"), dict):
            normalized["arguments"] = normalized["args"]

    if normalized.get("type") == "final" and "content" not in normalized:
        for key in ("message", "answer", "text"):
            value = normalized.get(key)
            if isinstance(value, str):
                normalized["content"] = value
                break

    return normalized


def _validate_action(data: Any, *, output: str, cleaned: str, stage: ParseStage) -> ParseOutcome:
    if not isinstance(data, dict):
        raise ParseError("Parsed JSON was not an object.", output, cleaned, 0)
    normalized = _normalize_aliases(data)
    try:
        action = _ACTION_ADAPTER.validate_python(normalized)
    except ValidationError as e:
        raise ParseError(
            "Output JSON did not match required action schema.",
            output,
            cleaned,
            len(e.errors()),
        ) from e
    return ParseOutcome(
        action=action,
        parse_stage=stage,
        parse_confidence="direct",
        validation_error_count=0,
    )


def parse_action_outcome_with_normalization(output: str) -> ParseOutcome:
    """
    Robust parser for local models that may emit:
    - fenced code blocks (```json ... ```)
    - JSON strings containing JSON
    - unquoted escaped JSON object text: {\\\"type\\\":...}
    """
    cleaned = _strip_code_fences(output)
    logger.debug("raw_output=%r", output)
    logger.debug("cleaned_output=%r", cleaned)

    # Step 2: normal parse
    try:
        data = json.loads(cleaned)
    except Exception:
        data = None

    if isinstance(data, dict):
        return _validate_action(data, output=output, cleaned=cleaned, stage="direct_json")

    # Step 3: parsed as string containing JSON
    if isinstance(data, str):
        try:
            inner = json.loads(data)
        except Exception as e:  # noqa: BLE001
            raise ParseError("Parsed a JSON string but it was not valid JSON.", output, cleaned, 0) from e
        return _validate_action(inner, output=output, cleaned=cleaned, stage="json_string")

    # Step 4: handle unquoted escaped JSON like: {\"type\":\"final\"}
    candidate = cleaned.lstrip()
    if candidate.startswith('{\\\"') or candidate.startswith('[\\\"'):
        try:
            # Ensure the wrapper itself is valid JSON string.
            safe = candidate.replace("\r", "").replace("\n", "\\n")
            fixed = json.loads(f"\"{safe}\"")
            inner = json.loads(fixed)
        except Exception as e:  # noqa: BLE001
            raise ParseError("Failed to unescape JSON-like output.", output, cleaned, 0) from e
        return _validate_action(inner, output=output, cleaned=cleaned, stage="escaped_json")

    raise ParseError("Could not parse a valid JSON object from model output.", output, cleaned, 0)


def parse_action_with_normalization(output: str) -> Action:
    # Backward-compatible helper for existing call sites/tests.
    return parse_action_outcome_with_normalization(output).action
