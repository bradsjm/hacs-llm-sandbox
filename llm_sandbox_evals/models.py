"""Model adapters for native tool-calling eval turns."""

import asyncio
import json
import re
import time
from collections.abc import Sequence
from typing import Literal, Protocol, cast

from custom_components.llm_sandbox.const import (
    TOOL_EXECUTE_HOME_CODE,
    TOOL_GET_HISTORY,
    TOOL_GET_LOGBOOK,
    TOOL_GET_STATISTICS,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from llm_sandbox_evals.schema import AgentStep, ToolCall

_ENTITY_ID_RE = re.compile(r"\b[a-z_]+\.[a-z0-9_]+\b")
_FIXED_END = "2026-06-29T12:00:00+00:00"
_FIXED_START = "2026-06-28T12:00:00+00:00"
_FIXED_TODAY_START = "2026-06-29T00:00:00+00:00"
_ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
_PYDANTIC_AI_RESPONSE_KEY = "_pydantic_ai_response"


class ModelAdapter(Protocol):
    """Protocol implemented by all model completion adapters."""

    async def respond(
        self, model_id: str, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> AgentStep:
        """Return one assistant step using native provider tool calling."""


class ModelResponseError(Exception):
    """Raised when a model provider fails before returning a usable assistant step."""

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        """Store a concise message plus provider details for stderr/artifacts."""
        super().__init__(message)
        self.detail = message if detail is None else detail


class StubAdapter:
    """Deterministic offline adapter used to validate the eval pipeline."""

    async def respond(
        self,
        model_id: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> AgentStep:
        """Emit one tool call, then a terminal answer that echoes the last tool result."""
        _ = (model_id, tools)
        last_tool_content = _last_tool_content(messages)
        user_request = _first_user_content(messages)

        tool_count = _tool_message_count(messages)
        # Branch boundary: selected multi-tool scenarios continue after prior tool output.
        if last_tool_content is not None:
            followup_calls = _stub_followup_calls(user_request, tool_count)
            if not followup_calls:
                text = _all_tool_content(messages) or last_tool_content
                return AgentStep(
                    tool_calls=(), text=text, assistant_message={"role": "assistant", "content": text}, raw=text
                )
            return _stub_step(followup_calls)

        return _stub_step(_stub_initial_calls(user_request))


def _stub_step(calls: tuple[ToolCall, ...]) -> AgentStep:
    """Return one deterministic assistant tool-call step."""
    tool_calls = [
        {
            "id": call.id,
            "type": "function",
            "function": {"name": call.tool_name, "arguments": json.dumps(call.tool_args, sort_keys=True)},
        }
        for call in calls
    ]
    message: dict[str, object] = {"role": "assistant", "content": "", "tool_calls": tool_calls}
    return AgentStep(tool_calls=calls, text="", assistant_message=message, raw=json.dumps(message, sort_keys=True))


def _stub_initial_calls(user_request: str) -> tuple[ToolCall, ...]:
    """Return the first deterministic tool call(s) for the stub adapter."""
    lowered = user_request.lower()
    if "sensor.living_room_temperature" in lowered:
        return (_call(1, TOOL_GET_HISTORY, {"entity_ids": ["sensor.living_room_temperature"], **_last_day_window()}),)
    if "summarize the living room temperature history" in lowered and "humidity hourly statistics" in lowered:
        return (
            _call(1, TOOL_GET_HISTORY, {"entity_ids": ["sensor.living_temp"], **_last_day_window()}),
            _call(
                2,
                TOOL_GET_STATISTICS,
                {"statistic_ids": ["sensor.bedroom_humidity"], "period": "hour", **_last_day_window()},
            ),
        )
    if "find the living room temperature sensor" in lowered:
        return (_call(1, TOOL_EXECUTE_HOME_CODE, {"code": 'result = states.get("sensor.living_temp")'}),)
    if "living room temperature has been above 25" in lowered:
        return (_call(1, TOOL_GET_HISTORY, {"entity_ids": ["sensor.living_temp"], **_last_day_window()}),)
    if "light.living last turn on" in lowered or "light.living last turned on" in lowered:
        return (
            _call(
                1,
                TOOL_GET_HISTORY,
                {"entity_ids": ["light.living"], "aggregate": "last_seen", "to_state": "on", **_last_day_window()},
            ),
        )
    if "living room light turned on today" in lowered:
        return (_call(1, TOOL_GET_LOGBOOK, {"entity_ids": ["light.living"], **_today_window()}),)
    if "living room light on right now" in lowered and "last change" in lowered:
        return (
            _call(1, TOOL_EXECUTE_HOME_CODE, {"code": 'result = states.get("light.living")'}),
            _call(2, TOOL_GET_LOGBOOK, {"entity_ids": ["light.living"], **_today_window()}),
        )
    if "light in this room" in lowered:
        return (_call(1, TOOL_EXECUTE_HOME_CODE, {"code": 'result = states.get("light.living")'}),)
    if "garage door opener" in lowered:
        return (_call(1, TOOL_EXECUTE_HOME_CODE, {"code": "result = states.entity_ids()"}),)
    if "outside temperature stayed below 80" in lowered:
        return (_call(1, TOOL_GET_HISTORY, {"entity_ids": ["sensor.tempest_temperature"], **_last_day_window()}),)

    tool_name = _select_stub_tool(user_request)
    return (_call(1, tool_name, _build_stub_tool_args(tool_name, user_request)),)


def _stub_followup_calls(user_request: str, tool_count: int) -> tuple[ToolCall, ...]:
    """Return the next deterministic tool call(s) after previous tool output."""
    lowered = user_request.lower()
    next_id = tool_count + 1
    if "sensor.living_room_temperature" in lowered and tool_count == 1:
        return (_call(next_id, TOOL_EXECUTE_HOME_CODE, {"code": 'result = states.get("sensor.living_temp")'}),)
    if "sensor.living_room_temperature" in lowered and tool_count == 2:
        return (_call(next_id, TOOL_GET_HISTORY, {"entity_ids": ["sensor.living_temp"], **_last_day_window()}),)
    if "find the living room temperature sensor" in lowered and tool_count == 1:
        return (_call(next_id, TOOL_GET_HISTORY, {"entity_ids": ["sensor.living_temp"], **_last_day_window()}),)
    if "living room temperature has been above 25" in lowered and tool_count == 1:
        return (_call(next_id, TOOL_EXECUTE_HOME_CODE, {"code": _fan_50_code("fan.living_fan")}),)
    if "living room light turned on today" in lowered and tool_count == 1:
        return (_call(next_id, TOOL_EXECUTE_HOME_CODE, {"code": _service_code("light", "turn_off", "light.living")}),)
    if "light in this room" in lowered and tool_count == 1:
        return (_call(next_id, TOOL_GET_LOGBOOK, {"entity_ids": ["light.living"], **_today_window()}),)
    if "outside temperature stayed below 80" in lowered and tool_count == 1:
        return (
            _call(
                next_id, TOOL_EXECUTE_HOME_CODE, {"code": _service_code("cover", "close_cover", "cover.office_blinds")}
            ),
        )
    return ()


def _call(index: int, tool_name: str, tool_args: dict[str, object]) -> ToolCall:
    """Build one deterministic stub tool call."""
    return ToolCall(id=f"stub-call-{index}", tool_name=tool_name, tool_args=tool_args)


def _last_day_window() -> dict[str, object]:
    """Return the fixed 24-hour eval recorder window."""
    return {"start": _FIXED_START, "end": _FIXED_END}


def _today_window() -> dict[str, object]:
    """Return the fixed same-day eval recorder window."""
    return {"start": _FIXED_TODAY_START, "end": _FIXED_END}


def _service_code(domain: str, service: str, entity_id: str) -> str:
    """Return minimal executable code that records a safe service call."""
    return (
        f'await hass.services.async_call("{domain}", "{service}", target={{"entity_id": "{entity_id}"}})\n'
        'result = "ok"'
    )


def _fan_50_code(entity_id: str) -> str:
    """Return minimal executable code that records a fan percentage service call."""
    return (
        'await hass.services.async_call("fan", "set_percentage", {"percentage": 50}, '
        f'target={{"entity_id": "{entity_id}"}})\nresult = "ok"'
    )


def _tool_message_count(messages: list[dict[str, object]]) -> int:
    """Return how many tool-result messages have already been appended."""
    return sum(1 for message in messages if message.get("role") == "tool")


def _all_tool_content(messages: list[dict[str, object]]) -> str | None:
    """Return all tool-result contents in order for deterministic stub summaries."""
    contents = [
        content
        for message in messages
        if message.get("role") == "tool" and isinstance((content := message.get("content")), str)
    ]
    if not contents:
        return None
    return "\n".join(contents)


class PydanticAIAdapter:
    """Adapter for real models routed through Pydantic AI direct model requests."""

    def __init__(self, reasoning_effort: str | None = None, *, model_timeout: float = 75.0) -> None:
        """Store model-call options forwarded to Pydantic AI."""
        self._reasoning_effort = reasoning_effort
        self._model_timeout = model_timeout

    async def respond(
        self,
        model_id: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> AgentStep:
        """Call Pydantic AI and normalize one assistant message into an AgentStep."""
        start = time.monotonic()
        try:
            from pydantic_ai.direct import model_request

            pydantic_messages = _to_pydantic_ai_messages(messages)
            tool_defs = _to_tool_definitions(tools)
            response = await asyncio.wait_for(
                model_request(
                    model_id,
                    pydantic_messages,
                    model_settings=reasoning_model_settings(model_id, self._reasoning_effort),
                    model_request_parameters=_request_parameters(tool_defs),
                ),
                timeout=self._model_timeout,
            )
            return _step_from_model_response(response)
        except TimeoutError as err:
            raise ModelResponseError(
                f"Pydantic AI completion timed out after {self._model_timeout:g} seconds",
                detail=_timeout_error_detail(
                    model_id=model_id,
                    model_timeout=self._model_timeout,
                    elapsed=time.monotonic() - start,
                ),
            ) from err
        except Exception as err:
            raise ModelResponseError(
                f"{type(err).__name__}: {err}",
                detail=_provider_error_detail(err, model_id=model_id),
            ) from err


def get_adapter(model_id: str, reasoning_effort: str | None = None, *, model_timeout: float = 75.0) -> ModelAdapter:
    """Return the adapter for a model identifier."""
    # Branch boundary: the stub adapter is a first-class offline model for keyless verification.
    if model_id == "stub":
        return StubAdapter()
    return PydanticAIAdapter(reasoning_effort=reasoning_effort, model_timeout=model_timeout)


def _to_pydantic_ai_messages(messages: list[dict[str, object]]) -> list[ModelMessage]:
    """Translate OpenAI-shaped harness messages into Pydantic AI messages."""
    translated: list[ModelMessage] = []
    tool_names_by_id: dict[str, str] = {}
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        if role == "system":
            translated.append(ModelRequest(parts=[SystemPromptPart(content=text)]))
            continue
        if role == "user":
            translated.append(ModelRequest(parts=[UserPromptPart(content=text)]))
            continue
        if role == "assistant":
            native_response = _native_response_from_message(message)
            # Branch boundary: real Pydantic AI turns replay the native response so reasoning/provider metadata survives.
            if native_response is not None:
                _remember_tool_names(native_response.parts, tool_names_by_id)
                translated.append(native_response)
                continue
            parts: list[TextPart | ToolCallPart] = []
            if text:
                parts.append(TextPart(content=text))
            for tool_call in _assistant_tool_call_parts(message.get("tool_calls"), tool_names_by_id):
                parts.append(tool_call)
            translated.append(ModelResponse(parts=parts))
            continue
        if role == "tool":
            tool_call_id = str(message.get("tool_call_id") or "")
            # Branch boundary: OpenAI tool messages omit the name; recover it from the prior assistant turn.
            tool_name = tool_names_by_id.get(tool_call_id, "unknown_tool")
            translated.append(
                ModelRequest(parts=[ToolReturnPart(tool_name=tool_name, content=text, tool_call_id=tool_call_id)])
            )
    return translated


def _native_response_from_message(message: dict[str, object]) -> ModelResponse | None:
    """Return the stored native Pydantic AI response for an assistant message, when present."""
    raw_response = message.get(_PYDANTIC_AI_RESPONSE_KEY)
    # Branch boundary: stub/tests and pre-native histories still use the OpenAI-shaped reconstruction path.
    if raw_response is None:
        return None
    restored = ModelMessagesTypeAdapter.validate_python([raw_response])
    response = restored[0]
    if not isinstance(response, ModelResponse):
        raise TypeError("stored Pydantic AI assistant message is not a ModelResponse")
    return response


def _remember_tool_names(parts: Sequence[ModelResponsePart], tool_names_by_id: dict[str, str]) -> None:
    """Remember native tool call names so following OpenAI-shaped tool returns can be named."""
    for part in parts:
        if isinstance(part, ToolCallPart) and part.tool_call_id is not None:
            tool_names_by_id[part.tool_call_id] = part.tool_name


def _assistant_tool_call_parts(raw_tool_calls: object, tool_names_by_id: dict[str, str]) -> list[ToolCallPart]:
    """Translate replayed assistant tool calls and remember ids for subsequent tool returns."""
    if not isinstance(raw_tool_calls, list):
        return []
    parts: list[ToolCallPart] = []
    for index, item in enumerate(raw_tool_calls):
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str):
            continue
        arguments = function.get("arguments")
        args = _json_object(arguments) if isinstance(arguments, str) else {}
        tool_call_id = str(item.get("id") or f"call-{index}")
        tool_names_by_id[tool_call_id] = name
        parts.append(ToolCallPart(tool_name=name, args=args, tool_call_id=tool_call_id))
    return parts


def _to_tool_definitions(tools: list[dict[str, object]]) -> list[ToolDefinition]:
    """Translate OpenAI function schemas into Pydantic AI tool definitions."""
    definitions: list[ToolDefinition] = []
    for tool in tools:
        function = tool.get("function")
        if tool.get("type") != "function" or not isinstance(function, dict):
            continue
        name = function.get("name")
        description = function.get("description")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, dict):
            continue
        definitions.append(
            ToolDefinition(
                name=name,
                description=description if isinstance(description, str) else None,
                parameters_json_schema=parameters,
            )
        )
    return definitions


def _request_parameters(tool_defs: list[ToolDefinition]) -> ModelRequestParameters:
    """Build Pydantic AI request parameters for harness-owned tool execution."""
    # Installed Pydantic AI 2.5.0 expects function_tools as list[ToolDefinition].
    return ModelRequestParameters(function_tools=tool_defs, allow_text_output=True)


def _step_from_model_response(response: ModelResponse) -> AgentStep:
    """Extract text and tool calls from a Pydantic AI model response."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    replay_tool_calls: list[dict[str, object]] = []
    for index, part in enumerate(response.parts):
        if isinstance(part, TextPart):
            text_parts.append(part.content)
            continue
        if isinstance(part, ToolCallPart):
            tool_args = part.args_as_dict()
            call_id = part.tool_call_id or f"call-{index}"
            tool_calls.append(ToolCall(id=call_id, tool_name=part.tool_name, tool_args=tool_args))
            replay_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": part.tool_name, "arguments": json.dumps(tool_args, sort_keys=True)},
                }
            )
    text = "".join(text_parts)
    native_response = _dump_native_response(response)
    assistant_message: dict[str, object] = {
        "role": "assistant",
        "content": text,
        _PYDANTIC_AI_RESPONSE_KEY: native_response,
    }
    # Branch boundary: replay tool calls only when the provider requested tool execution.
    if replay_tool_calls:
        assistant_message["tool_calls"] = replay_tool_calls
    return AgentStep(
        tool_calls=tuple(tool_calls),
        text=text,
        assistant_message=assistant_message,
        raw=json.dumps(native_response, sort_keys=True),
    )


def _dump_native_response(response: ModelResponse) -> dict[str, object]:
    """Serialize one Pydantic AI response for native next-turn replay and JSON artifacts."""
    dumped = ModelMessagesTypeAdapter.dump_python([response], mode="json")[0]
    if not isinstance(dumped, dict):
        raise TypeError("serialized Pydantic AI response is not a JSON object")
    return dumped


def reasoning_model_settings(model_id: str, reasoning_effort: str | None) -> ModelSettings | None:
    """Return provider-specific reasoning settings for Pydantic AI model ids."""
    # Branch boundary: no reasoning requested preserves the old deterministic eval default.
    if reasoning_effort is None:
        return ModelSettings(temperature=0.0)
    # Branch boundary: explicit no-reasoning keeps deterministic decoding for compatible providers.
    if reasoning_effort == "none":
        return ModelSettings(temperature=0.0)
    effort = cast(_ReasoningEffort, reasoning_effort)
    # Branch boundary: OpenRouter exposes an effort-shaped reasoning setting.
    if model_id.startswith("openrouter:"):
        from pydantic_ai.models.openrouter import OpenRouterModelSettings

        return OpenRouterModelSettings(openrouter_reasoning={"effort": effort})
    # Branch boundary: OpenAI Responses exposes a native reasoning effort setting.
    if model_id.startswith(("openai:", "openai-chat:")):
        from pydantic_ai.models.openai import OpenAIResponsesModelSettings

        return OpenAIResponsesModelSettings(openai_reasoning_effort=effort)
    return None


def _provider_error_detail(err: BaseException, *, model_id: str) -> str:
    """Return useful provider diagnostics without dumping request payloads."""
    lines = [f"requested model: {model_id}", _exception_line(err)]
    cause = err.__cause__ or err.__context__
    while cause is not None:
        lines.append("caused by: " + _exception_line(cause))
        cause = cause.__cause__ or cause.__context__
    return "\n".join(lines)


def _timeout_error_detail(*, model_id: str, model_timeout: float, elapsed: float) -> str:
    """Return an actionable diagnostic when the provider never returns a response."""
    return "\n".join(
        (
            f"requested model: {model_id}",
            f"model generation exceeded the eval timeout after {elapsed:.1f}s",
            f"eval model timeout: {model_timeout:g}s",
            "the provider did not return an HTTP status or response body before this timeout",
            "a provider's global status page can still be healthy when an individual generation queues or runs this long",
            "increase --model-timeout for slow/free models, lower --concurrency, or use the paid/non-free model id if you need reliable eval throughput",
        )
    )


def _exception_line(err: BaseException) -> str:
    """Format one exception with its fully qualified type."""
    return f"{type(err).__module__}.{type(err).__name__}: {err}"


def _limit_detail(value: str) -> str:
    """Bound provider response details so one failing model does not flood stderr."""
    compact = " ".join(value.split())
    if len(compact) <= 1000:
        return compact
    return compact[:1000] + "..."


def _json_object(value: str) -> dict[str, object]:
    """Decode a JSON object string for tool arguments."""
    try:
        decoded = json.loads(value) if value else {}
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _first_user_content(messages: list[dict[str, object]]) -> str:
    """Return the first user message content for deterministic stub routing."""
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def _last_tool_content(messages: list[dict[str, object]]) -> str | None:
    """Return the latest tool-result content if the loop has run a tool."""
    for message in reversed(messages):
        if message.get("role") == "tool":
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return None


def _select_stub_tool(user_request: str) -> str:
    """Choose a tool name with deterministic keyword matching."""
    lowered = user_request.lower()
    if "logbook" in lowered or "happened" in lowered:
        return TOOL_GET_LOGBOOK
    if "statistic" in lowered or "hourly" in lowered or "average over" in lowered:
        return TOOL_GET_STATISTICS
    if "history" in lowered or "last 24" in lowered or "over the last" in lowered:
        return TOOL_GET_HISTORY
    return TOOL_EXECUTE_HOME_CODE


def _build_stub_tool_args(tool_name: str, user_request: str) -> dict[str, object]:
    """Build deterministic, runnable tool arguments for the selected stub tool."""
    if tool_name == TOOL_EXECUTE_HOME_CODE:
        return {"code": "states.entity_ids()"}

    entity_ids = _ENTITY_ID_RE.findall(user_request)
    # Branch boundary: explicit ids exercise direct visibility; otherwise selectors exercise resolver logic.
    if tool_name == TOOL_GET_STATISTICS:
        args: dict[str, object] = {"start": _FIXED_START, "end": _FIXED_END, "period": "hour"}
        if entity_ids:
            args["statistic_ids"] = entity_ids[:1]
        else:
            args["domain"] = "sensor"
        return args

    args = {"start": _FIXED_START, "end": _FIXED_END}
    if entity_ids:
        args["entity_ids"] = entity_ids[:1]
    else:
        args["domain"] = "light" if tool_name == TOOL_GET_LOGBOOK else "sensor"
    return args
