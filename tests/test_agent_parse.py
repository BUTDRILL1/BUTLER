from __future__ import annotations

import pytest

from butler.agent.parsing import (
    ParseError,
    parse_action_outcome_with_normalization,
    parse_action_with_normalization,
)
from butler.agent.schema import FinalAction, ToolCallAction


def test_parse_action_final() -> None:
    out = parse_action_outcome_with_normalization('{"type":"final","content":"hi"}')
    assert isinstance(out.action, FinalAction)
    assert out.action.content == "hi"
    assert out.parse_stage == "direct_json"
    assert out.parse_confidence == "direct"


def test_parse_action_tool_call_with_codefence() -> None:
    out = parse_action_outcome_with_normalization(
        '```json\n{"type":"tool_call","name":"system.now","arguments":{}}\n```'
    )
    assert isinstance(out.action, ToolCallAction)
    assert out.action.name == "system.now"
    assert out.parse_stage == "direct_json"


def test_parse_action_rejects_non_json() -> None:
    with pytest.raises(ParseError):
        parse_action_outcome_with_normalization("hello")


def test_parse_action_unquoted_escaped_json_object() -> None:
    out = parse_action_outcome_with_normalization('{\\\"type\\\":\\\"final\\\",\\\"content\\\":\\\"hi\\\"}')
    assert isinstance(out.action, FinalAction)
    assert out.parse_stage == "escaped_json"


def test_parse_action_json_string_containing_json() -> None:
    out = parse_action_outcome_with_normalization('"{\\"type\\":\\"final\\",\\"content\\":\\"hi\\"}"')
    assert isinstance(out.action, FinalAction)
    assert out.parse_stage == "json_string"


def test_parse_alias_mapping_action_and_args() -> None:
    out = parse_action_outcome_with_normalization('{"action":"tool_call","name":"system.now","args":{}}')
    assert isinstance(out.action, ToolCallAction)
    assert out.action.arguments == {}


def test_parse_alias_mapping_final_content() -> None:
    a = parse_action_with_normalization('{"action":"final","message":"hello"}')
    assert isinstance(a, FinalAction)
    assert a.content == "hello"
