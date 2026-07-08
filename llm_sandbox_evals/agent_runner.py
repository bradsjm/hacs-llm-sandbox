"""Pydantic AI agent wiring for evals over production tool cores."""

import json
import re
from collections.abc import Mapping
from typing import Literal, Protocol, cast

from custom_components.llm_sandbox.const import (
    TOOL_EXECUTE_HOME_CODE,
    TOOL_GET_HISTORY,
    TOOL_GET_LOGBOOK,
    TOOL_GET_STATISTICS,
)
from custom_components.llm_sandbox.llm_api.errors import setup_error_payload, tool_error_from_exception
from custom_components.llm_sandbox.llm_api.prompts import (
    compose_system_prompt,
    render_home_inventory,
    render_request_location,
)
from custom_components.llm_sandbox.llm_api.tools.code import ExecuteHomeCodeTool
from custom_components.llm_sandbox.llm_api.tools.recorder import (
    GetHistoryTool,
    GetLogbookTool,
    GetStatisticsTool,
    recorder_error_envelope,
)
from homeassistant.helpers import llm
from homeassistant.util.json import JsonObjectType
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings
from voluptuous_openapi import convert

from llm_sandbox_evals.runtime import EvalRuntime

_ENTITY_ID_RE = re.compile(r"\b[a-z_]+\.[a-z0-9_]+\b")
_FIXED_END = "2026-06-29T12:00:00+00:00"
_FIXED_START = "2026-06-28T12:00:00+00:00"
_FIXED_TODAY_START = "2026-06-29T00:00:00+00:00"
type _ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class _EvalTool(Protocol):
    name: str
    description: str

    def _normalize_args(self, args: Mapping[str, object]) -> dict[str, object]: ...


def build_agent(runtime: EvalRuntime, model_id: str) -> Agent[EvalRuntime, str]:
    """Build a Pydantic AI agent with production schemas and eval deps."""
    return Agent(
        model=make_model(model_id),
        tools=build_agent_tools(runtime),
        system_prompt=render_eval_system_prompt(runtime),
        output_type=str,
        deps_type=EvalRuntime,
        name="llm_sandbox_eval",
    )


def build_agent_tools(runtime: EvalRuntime) -> list[Tool[EvalRuntime]]:
    """Build executable Pydantic AI tools backed by production tool cores."""
    tools: list[Tool[EvalRuntime]] = [_make_code_tool(runtime.code_tool)]
    for tool in runtime.recorder_tools:
        # Branch boundary: production omits get_logbook when logbook is unavailable.
        if isinstance(tool, GetLogbookTool) and not runtime.recorder_source.logbook_available:
            continue
        tools.append(_make_recorder_tool(tool))
    return tools


def _make_recorder_tool(tool: GetHistoryTool | GetStatisticsTool | GetLogbookTool) -> Tool[EvalRuntime]:
    """Return one pydantic-ai Tool backed by _RecorderTool.run_query."""
    json_schema = convert(tool.parameters)

    async def execute(ctx: RunContext[EvalRuntime], **kwargs: object) -> JsonObjectType:
        validation = _validate_recorder_tool(tool, kwargs)
        if validation.error is not None:
            return validation.error
        return await tool.run_query(ctx.deps.snapshot, validation.data, ctx.deps.recorder_source)

    return Tool.from_schema(
        execute,
        name=tool.name,
        description=tool.description,
        json_schema=json_schema,
        takes_ctx=True,
    )


def _make_code_tool(tool: ExecuteHomeCodeTool) -> Tool[EvalRuntime]:
    """Return one pydantic-ai Tool backed by ExecuteHomeCodeTool.run_execute."""
    json_schema = convert(tool.parameters)

    async def execute(ctx: RunContext[EvalRuntime], **kwargs: object) -> JsonObjectType:
        validation = _validate_code_tool(tool, kwargs)
        if validation.error is not None:
            return validation.error
        runtime = ctx.deps.runtime_context_factory()
        llm_context = llm.LLMContext(
            ctx.deps.case.llm_context.platform,
            None,
            ctx.deps.case.llm_context.language,
            None,
            ctx.deps.case.llm_context.device_id,
        )
        return await tool.run_execute(ctx.deps.snapshot, validation.data, llm_context, runtime)

    return Tool.from_schema(
        execute,
        name=tool.name,
        description=tool.description,
        json_schema=json_schema,
        takes_ctx=True,
    )


class _ValidationResult:
    """Validated args or a production-shaped invalid-input envelope."""

    def __init__(self, data: dict[str, object], error: JsonObjectType | None) -> None:
        self.data = data
        self.error = error


def _validate_recorder_tool(tool: _EvalTool, kwargs: dict[str, object]) -> _ValidationResult:
    """Validate recorder args in production ordering."""
    try:
        return _ValidationResult(cast(dict[str, object], tool.parameters(tool._normalize_args(kwargs))), None)  # type: ignore[attr-defined]
    except Exception as err:
        mapped = tool_error_from_exception(err)
        if mapped is None:
            raise
        return _ValidationResult({}, recorder_error_envelope(*mapped))


def _validate_code_tool(tool: ExecuteHomeCodeTool, kwargs: dict[str, object]) -> _ValidationResult:
    """Validate execute_home_code args in production ordering."""
    try:
        return _ValidationResult(cast(dict[str, object], tool.parameters(kwargs)), None)
    except Exception as err:
        mapped = tool_error_from_exception(err)
        if mapped is None:
            raise
        key, placeholders = mapped
        return _ValidationResult({}, cast(JsonObjectType, setup_error_payload(key, placeholders)))


def render_eval_system_prompt(runtime: EvalRuntime) -> str:
    """Render the byte-identical production system prompt for a fixture snapshot."""
    inventory_section = render_home_inventory(
        runtime.snapshot,
        recorder_available=True,
        logbook_available=runtime.recorder_source.logbook_available,
    )
    return compose_system_prompt(
        runtime.candidate.api_prompt,
        runtime.case.actions_enabled,
        location_section=_eval_location_section(runtime),
        inventory_section=inventory_section,
    )


def _eval_location_section(runtime: EvalRuntime) -> str | None:
    """Render the request-location prompt section from frozen snapshot records."""
    device_id = runtime.case.llm_context.device_id
    if device_id is None:
        return None
    device = runtime.snapshot.devices.get(device_id)
    area_id = device.area_id if device is not None else None
    area = runtime.snapshot.areas.get(area_id) if area_id is not None else None
    floor_id = area.floor_id if area is not None else None
    floor = runtime.snapshot.floors.get(floor_id) if floor_id is not None else None
    return render_request_location(
        device_id,
        area.id if area is not None else None,
        area.name if area is not None else None,
        floor.floor_id if floor is not None else None,
        floor.name if floor is not None else None,
    )


def make_model(model_id: str) -> Model:
    """Return the Pydantic AI model for an eval model id."""
    if model_id == "stub":
        return stub_function_model()
    return infer_model(model_id)


def stub_function_model() -> FunctionModel:
    """Return the deterministic keyless FunctionModel used for CI pipeline validation."""
    return FunctionModel(_stub_respond, model_name="stub")


async def _stub_respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Deterministically route the first user request to tool calls or terminal text."""
    _ = info
    user_request = _first_user_content(messages)
    tool_count = _tool_return_count(messages)
    if _last_tool_content(messages) is not None:
        followup_calls = _stub_followup_calls(user_request, tool_count)
        if followup_calls:
            return ModelResponse(
                parts=[
                    _tool_call_part(name, args, tool_count + index)
                    for index, (name, args) in enumerate(followup_calls, start=1)
                ]
            )
        return ModelResponse(parts=[TextPart(content=_all_tool_content(messages) or "")])
    return ModelResponse(
        parts=[
            _tool_call_part(name, args, index)
            for index, (name, args) in enumerate(_stub_initial_calls(user_request), start=1)
        ]
    )


def _tool_call_part(tool_name: str, tool_args: dict[str, object], index: int) -> ToolCallPart:
    """Build one deterministic Pydantic AI tool call part."""
    return ToolCallPart(tool_name=tool_name, args=tool_args, tool_call_id=f"stub-call-{index}")


def _stub_initial_calls(user_request: str) -> tuple[tuple[str, dict[str, object]], ...]:
    """Return the first deterministic tool call(s) for the stub model."""
    lowered = user_request.lower()
    if "sensor.living_room_temperature" in lowered:
        return ((TOOL_GET_HISTORY, {"entity_ids": ["sensor.living_room_temperature"], **_last_day_window()}),)
    if "summarize the living room temperature history" in lowered and "humidity hourly statistics" in lowered:
        return (
            (TOOL_GET_HISTORY, {"entity_ids": ["sensor.living_temp"], **_last_day_window()}),
            (
                TOOL_GET_STATISTICS,
                {"statistic_ids": ["sensor.bedroom_humidity"], "period": "hour", **_last_day_window()},
            ),
        )
    if "find the living room temperature sensor" in lowered:
        return ((TOOL_EXECUTE_HOME_CODE, {"code": 'result = states.get("sensor.living_temp")'}),)
    if "living room temperature has been above 25" in lowered:
        return ((TOOL_GET_HISTORY, {"entity_ids": ["sensor.living_temp"], **_last_day_window()}),)
    if "light.living last turn on" in lowered or "light.living last turned on" in lowered:
        return (
            (
                TOOL_GET_HISTORY,
                {"entity_ids": ["light.living"], "aggregate": "last_seen", "to_state": "on", **_last_day_window()},
            ),
        )
    if "living room light turned on today" in lowered:
        return ((TOOL_GET_LOGBOOK, {"entity_ids": ["light.living"], **_today_window()}),)
    if "living room light on right now" in lowered and "last change" in lowered:
        return (
            (TOOL_EXECUTE_HOME_CODE, {"code": 'result = states.get("light.living")'}),
            (TOOL_GET_LOGBOOK, {"entity_ids": ["light.living"], **_today_window()}),
        )
    if "light in this room" in lowered:
        return ((TOOL_EXECUTE_HOME_CODE, {"code": 'result = states.get("light.living")'}),)
    if "garage door opener" in lowered:
        return ((TOOL_EXECUTE_HOME_CODE, {"code": "result = states.entity_ids()"}),)
    if "outside temperature stayed below 80" in lowered:
        return ((TOOL_GET_HISTORY, {"entity_ids": ["sensor.tempest_temperature"], **_last_day_window()}),)
    # Branch boundary: state-read of the living room temperature reads the entity so
    # the observed value surfaces in the tool return + final answer. Exclude history
    # and threshold phrasings, which are handled by their own routes or the recorder
    # fallback below.
    if "temperature" in lowered and "living room" in lowered and "history" not in lowered and "above" not in lowered:
        return ((TOOL_EXECUTE_HOME_CODE, {"code": 'result = states.get("sensor.living_temp")'}),)

    tool_name = _select_stub_tool(user_request)
    return ((tool_name, _build_stub_tool_args(tool_name, user_request)),)


def _stub_followup_calls(user_request: str, tool_count: int) -> tuple[tuple[str, dict[str, object]], ...]:
    """Return the next deterministic tool call(s) after previous tool output."""
    lowered = user_request.lower()
    if "sensor.living_room_temperature" in lowered and tool_count == 1:
        return ((TOOL_EXECUTE_HOME_CODE, {"code": 'result = states.get("sensor.living_temp")'}),)
    if "sensor.living_room_temperature" in lowered and tool_count == 2:
        return ((TOOL_GET_HISTORY, {"entity_ids": ["sensor.living_temp"], **_last_day_window()}),)
    if "find the living room temperature sensor" in lowered and tool_count == 1:
        return ((TOOL_GET_HISTORY, {"entity_ids": ["sensor.living_temp"], **_last_day_window()}),)
    if "living room temperature has been above 25" in lowered and tool_count == 1:
        return ((TOOL_EXECUTE_HOME_CODE, {"code": _fan_50_code("fan.living_fan")}),)
    if "living room light turned on today" in lowered and tool_count == 1:
        return ((TOOL_EXECUTE_HOME_CODE, {"code": _service_code("light", "turn_off", "light.living")}),)
    if "light in this room" in lowered and tool_count == 1:
        return ((TOOL_GET_LOGBOOK, {"entity_ids": ["light.living"], **_today_window()}),)
    if "outside temperature stayed below 80" in lowered and tool_count == 1:
        return ((TOOL_EXECUTE_HOME_CODE, {"code": _service_code("cover", "close_cover", "cover.office_blinds")}),)
    return ()


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


def _first_user_content(messages: list[ModelMessage]) -> str:
    """Return the first user message content for deterministic stub routing."""
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    return part.content if isinstance(part.content, str) else ""
    return ""


def _tool_return_count(messages: list[ModelMessage]) -> int:
    """Return how many tool returns the model has seen."""
    return sum(
        1
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    )


def _last_tool_content(messages: list[ModelMessage]) -> str | None:
    """Return the latest tool-return JSON content if a tool has run."""
    for message in reversed(messages):
        if not isinstance(message, ModelRequest):
            continue
        for part in reversed(message.parts):
            if isinstance(part, ToolReturnPart):
                return json.dumps(part.content, sort_keys=True, default=str)
    return None


def _all_tool_content(messages: list[ModelMessage]) -> str | None:
    """Return all tool-return JSON content in order for deterministic stub summaries."""
    contents = [
        json.dumps(part.content, sort_keys=True, default=str)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    return "\n".join(contents) if contents else None


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
        return {"code": "result = states.entity_ids()"}

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


def build_model_settings(
    model_id: str,
    *,
    temperature: float | None,
    reasoning_effort: str | None,
) -> ModelSettings | None:
    """Return provider model settings containing only values explicitly provided.

    Never defaults sampling parameters (e.g. ``temperature=0.0``). Reasoning-capable
    OpenAI/OpenRouter models that cannot disable reasoning warn and drop sampling
    params whenever one is present, so a default temperature surfaced that warning on
    every run; only forward what the caller asked for and leave the rest to the provider.
    """
    reasoning_value = _resolve_reasoning_value(reasoning_effort)
    # Branch boundary: an explicit reasoning effort selects the provider's reasoning setting.
    if reasoning_value is not None:
        # Branch boundary: OpenRouter exposes an effort-shaped reasoning setting.
        if model_id.startswith("openrouter:"):
            from pydantic_ai.models.openrouter import OpenRouterModelSettings

            return OpenRouterModelSettings(
                openrouter_reasoning={"effort": reasoning_value},
                **({"temperature": temperature} if temperature is not None else {}),
            )
        # Branch boundary: OpenAI Responses exposes a native reasoning effort setting.
        if model_id.startswith(("openai:", "openai-chat:")):
            from pydantic_ai.models.openai import OpenAIResponsesModelSettings

            return OpenAIResponsesModelSettings(
                openai_reasoning_effort=reasoning_value,
                **({"temperature": temperature} if temperature is not None else {}),
            )
    # Branch boundary: no active reasoning — forward temperature only when explicitly provided.
    if temperature is not None:
        return ModelSettings(temperature=temperature)
    return None


def _resolve_reasoning_value(reasoning_effort: str | None) -> _ReasoningEffort | None:
    """Map a CLI reasoning effort to a provider value, treating 'none' as not requested."""
    if reasoning_effort is None or reasoning_effort == "none":
        return None
    return cast(_ReasoningEffort, reasoning_effort)
