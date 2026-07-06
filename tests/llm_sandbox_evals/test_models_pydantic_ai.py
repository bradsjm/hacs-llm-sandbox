import json

import pytest
from llm_sandbox_evals.models import (
    _step_from_model_response,
    _to_pydantic_ai_messages,
    _to_tool_definitions,
    reasoning_model_settings,
)
from llm_sandbox_evals.prompts import baseline_candidate, function_schemas
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)


def test_to_tool_definitions_builds_pydantic_ai_tool_definitions() -> None:
    schemas = function_schemas(baseline_candidate())

    definitions = _to_tool_definitions(schemas)

    assert [definition.name for definition in definitions] == [schema["function"]["name"] for schema in schemas]
    assert [definition.description for definition in definitions] == [
        schema["function"]["description"] for schema in schemas
    ]
    assert [definition.parameters_json_schema for definition in definitions] == [
        schema["function"]["parameters"] for schema in schemas
    ]


def test_to_pydantic_ai_messages_round_trips_system_user_assistant_tool_result() -> None:
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "execute_home_code", "arguments": '{"code": "x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": '{"ok": true}'},
    ]

    translated = _to_pydantic_ai_messages(messages)

    assert isinstance(translated[0], ModelRequest)
    assert isinstance(translated[0].parts[0], SystemPromptPart)
    assert translated[0].parts[0].content == "system"
    assert isinstance(translated[1], ModelRequest)
    assert isinstance(translated[1].parts[0], UserPromptPart)
    assert translated[1].parts[0].content == "user"
    assert isinstance(translated[2], ModelResponse)
    assert isinstance(translated[2].parts[0], ToolCallPart)
    assert translated[2].parts[0].tool_name == "execute_home_code"
    assert translated[2].parts[0].args == {"code": "x"}
    assert isinstance(translated[3], ModelRequest)
    assert isinstance(translated[3].parts[0], ToolReturnPart)
    assert translated[3].parts[0].tool_name == "execute_home_code"
    assert translated[3].parts[0].tool_call_id == "call-1"


def test_step_from_model_response_extracts_text_and_tool_calls() -> None:
    response = ModelResponse(
        parts=[
            TextPart(content="answer"),
            ToolCallPart(tool_name="execute_home_code", args={"code": "x"}, tool_call_id="call-1"),
        ],
        provider_response_id="response-1",
    )

    step = _step_from_model_response(response)

    assert step.text == "answer"
    assert len(step.tool_calls) == 1
    assert step.tool_calls[0].id == "call-1"
    assert step.tool_calls[0].tool_name == "execute_home_code"
    assert step.tool_calls[0].tool_args == {"code": "x"}
    assert step.assistant_message["role"] == "assistant"
    assert step.assistant_message["content"] == "answer"
    assert step.assistant_message["tool_calls"] == [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "execute_home_code", "arguments": json.dumps({"code": "x"}, sort_keys=True)},
        }
    ]
    assert step.assistant_message["_pydantic_ai_response"]["provider_response_id"] == "response-1"


def test_to_pydantic_ai_messages_replays_native_assistant_response_parts() -> None:
    response = ModelResponse(
        parts=[
            ThinkingPart(content="reasoning", id="think-1", signature="sig", provider_details={"provider": "detail"}),
            ToolCallPart(tool_name="execute_home_code", args={"code": "x"}, tool_call_id="call-1"),
        ],
        provider_response_id="response-1",
        provider_details={"trace": "kept"},
    )
    step = _step_from_model_response(response)
    messages = [
        step.assistant_message,
        {"role": "tool", "tool_call_id": "call-1", "content": '{"ok": true}'},
    ]

    translated = _to_pydantic_ai_messages(messages)

    assert isinstance(translated[0], ModelResponse)
    assert translated[0].provider_response_id == "response-1"
    assert translated[0].provider_details == {"trace": "kept"}
    assert isinstance(translated[0].parts[0], ThinkingPart)
    assert translated[0].parts[0].id == "think-1"
    assert translated[0].parts[0].signature == "sig"
    assert translated[0].parts[0].provider_details == {"provider": "detail"}
    assert isinstance(translated[1], ModelRequest)
    assert isinstance(translated[1].parts[0], ToolReturnPart)
    assert translated[1].parts[0].tool_name == "execute_home_code"


def test_reasoning_model_settings_dispatches_openrouter() -> None:
    settings = reasoning_model_settings("openrouter:openai/gpt-4o-mini", "high")

    assert settings == {"openrouter_reasoning": {"effort": "high"}}


def test_reasoning_model_settings_dispatches_openai() -> None:
    settings = reasoning_model_settings("openai:gpt-4o-mini", "high")

    assert settings == {"openai_reasoning_effort": "high"}


@pytest.mark.parametrize(
    "reasoning_effort",
    [
        pytest.param(None, id="omitted"),
        pytest.param("none", id="explicit-none"),
    ],
)
def test_reasoning_model_settings_uses_deterministic_temperature_without_reasoning(
    reasoning_effort: str | None,
) -> None:
    settings = reasoning_model_settings("openai:gpt-4o-mini", reasoning_effort)

    assert settings == {"temperature": 0.0}


@pytest.mark.parametrize(
    ("model_id", "reasoning_effort"),
    [
        pytest.param("anthropic:claude-haiku-4-5", "high", id="anthropic-unsupported"),
    ],
)
def test_reasoning_model_settings_returns_none_without_supported_reasoning(
    model_id: str,
    reasoning_effort: str | None,
) -> None:
    settings = reasoning_model_settings(model_id, reasoning_effort)

    assert settings is None
