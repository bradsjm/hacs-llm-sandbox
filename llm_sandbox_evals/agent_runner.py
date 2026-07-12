"""Pydantic AI agent wiring for evals over production tool cores."""

from collections.abc import Mapping
import json
import re
from typing import Literal, Protocol, cast

from custom_components.llm_sandbox.const import (
    TOOL_EXECUTE_HOME_CODE,
    TOOL_GET_AUTOMATION,
    TOOL_GET_HISTORY,
    TOOL_GET_LOGBOOK,
    TOOL_GET_STATISTICS,
)
from custom_components.llm_sandbox.llm_api.errors import (
    setup_error_payload,
    tool_error_envelope,
    tool_error_from_exception,
)
from custom_components.llm_sandbox.llm_api.prompts import (
    compose_system_prompt,
    render_home_inventory,
    render_request_location,
    render_tool_capabilities,
)
from custom_components.llm_sandbox.llm_api.tools.automation import GetAutomationTool
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
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings
from voluptuous_openapi import convert

from llm_sandbox_evals.runtime import EvalRuntime
from llm_sandbox_evals.schema import AggregateClaim, AnswerClaim, EvalAnswer, EventClaim, ValueClaim

_ENTITY_ID_RE = re.compile(r"\b[a-z_]+\.[a-z0-9_]+\b")
_FIXED_END = "2026-06-29T12:00:00+00:00"
_FIXED_START = "2026-06-28T12:00:00+00:00"
_FIXED_TODAY_START = "2026-06-29T00:00:00+00:00"
type _ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class _EvalTool(Protocol):
    name: str
    description: str

    def _normalize_args(self, args: Mapping[str, object]) -> dict[str, object]: ...


def build_agent(runtime: EvalRuntime, model_id: str) -> Agent[EvalRuntime, EvalAnswer]:
    """Build a Pydantic AI agent with production schemas and eval deps."""
    tools = build_agent_tools(runtime)
    return Agent(
        model=make_model(model_id),
        tools=tools,
        system_prompt=render_eval_system_prompt(runtime, tools),
        output_type=EvalAnswer,
        deps_type=EvalRuntime,
        name="llm_sandbox_eval",
    )


def build_agent_tools(runtime: EvalRuntime) -> list[Tool[EvalRuntime]]:
    """Build executable Pydantic AI tools backed by production tool cores."""
    descriptions = {
        TOOL_GET_HISTORY: runtime.candidate.get_history_description,
        TOOL_GET_STATISTICS: runtime.candidate.get_statistics_description,
        TOOL_GET_LOGBOOK: runtime.candidate.get_logbook_description,
        TOOL_GET_AUTOMATION: runtime.candidate.get_automation_description,
    }
    tools: list[Tool[EvalRuntime]] = [
        _make_code_tool(runtime.code_tool, runtime.candidate.execute_home_code_description)
    ]
    tools.append(_make_automation_tool(runtime.automation_tool, descriptions[TOOL_GET_AUTOMATION]))
    for tool in runtime.recorder_tools:
        # Branch boundary: production omits get_logbook when logbook is unavailable.
        if isinstance(tool, GetLogbookTool) and not runtime.recorder_source.logbook_available:
            continue
        tools.append(_make_recorder_tool(tool, descriptions[tool.name]))
    return tools


def _make_automation_tool(tool: GetAutomationTool, description: str) -> Tool[EvalRuntime]:
    """Return a Pydantic AI tool backed by the production automation query core."""
    json_schema = convert(tool.parameters)

    async def execute(ctx: RunContext[EvalRuntime], **kwargs: object) -> JsonObjectType:
        _notify_tool_boundary(ctx.deps, tool.name, started=True)
        try:
            validation = _validate_automation_tool(tool, kwargs)
            if validation.error is not None:
                return validation.error
            return await tool.run_query(validation.data, ctx.deps.automation_source)
        finally:
            _notify_tool_boundary(ctx.deps, tool.name, started=False)

    return Tool.from_schema(
        execute,
        name=tool.name,
        description=description,
        json_schema=json_schema,
        takes_ctx=True,
    )


def _make_recorder_tool(
    tool: GetHistoryTool | GetStatisticsTool | GetLogbookTool, description: str
) -> Tool[EvalRuntime]:
    """Return one pydantic-ai Tool backed by _RecorderTool.run_query."""
    json_schema = convert(tool.parameters)

    async def execute(ctx: RunContext[EvalRuntime], **kwargs: object) -> JsonObjectType:
        _notify_tool_boundary(ctx.deps, tool.name, started=True)
        try:
            validation = _validate_recorder_tool(tool, kwargs)
            if validation.error is not None:
                return validation.error
            return await tool.run_query(ctx.deps.snapshot, validation.data, ctx.deps.recorder_source)
        finally:
            _notify_tool_boundary(ctx.deps, tool.name, started=False)

    return Tool.from_schema(
        execute,
        name=tool.name,
        description=description,
        json_schema=json_schema,
        takes_ctx=True,
    )


def _make_code_tool(tool: ExecuteHomeCodeTool, description: str) -> Tool[EvalRuntime]:
    """Return one pydantic-ai Tool backed by ExecuteHomeCodeTool.run_execute."""
    json_schema = convert(tool.parameters)

    async def execute(ctx: RunContext[EvalRuntime], **kwargs: object) -> JsonObjectType:
        _notify_tool_boundary(ctx.deps, tool.name, started=True)
        try:
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
        finally:
            _notify_tool_boundary(ctx.deps, tool.name, started=False)

    return Tool.from_schema(
        execute,
        name=tool.name,
        description=description,
        json_schema=json_schema,
        takes_ctx=True,
    )


class _ValidationResult:
    """Validated args or a production-shaped invalid-input envelope."""

    def __init__(self, data: dict[str, object], error: JsonObjectType | None) -> None:
        self.data = data
        self.error = error


def _notify_tool_boundary(runtime: EvalRuntime, tool_name: str, *, started: bool) -> None:
    """Notify the optional eval observer without affecting tool semantics."""
    if runtime.on_tool_boundary is None:
        return
    try:
        runtime.on_tool_boundary(tool_name, started)
    except Exception:  # noqa: BLE001 - terminal observation must not change a tool result.
        return


def _validate_recorder_tool(tool: _EvalTool, kwargs: dict[str, object]) -> _ValidationResult:
    """Validate recorder args in production ordering."""
    try:
        if "cursor" in kwargs:
            data = cast(dict[str, object], tool.parameters({"cursor": kwargs["cursor"]}))  # type: ignore[attr-defined]
            # The private eval wrapper retains the already-resolved cursor scope for
            # production run_query; only the opaque cursor is used for continuation state.
            if isinstance(kwargs.get("entity_ids"), list):
                data["entity_ids"] = kwargs["entity_ids"]
            return _ValidationResult(data, None)
        return _ValidationResult(cast(dict[str, object], tool.parameters(tool._normalize_args(kwargs))), None)  # type: ignore[attr-defined]
    except Exception as err:
        mapped = tool_error_from_exception(err)
        if mapped is None:
            raise
        return _ValidationResult({}, recorder_error_envelope(*mapped))


def _validate_automation_tool(tool: GetAutomationTool, kwargs: dict[str, object]) -> _ValidationResult:
    """Validate automation args using the direct tool's normalizer and envelope."""
    try:
        normalized = tool._normalize_args(kwargs)
        if "cursor" in normalized and len(normalized) != 1:
            raise ValueError("cursor must be the only non-empty argument")
        data = cast(dict[str, object], tool.parameters(normalized))
        tool._validate_query_data(data)
        return _ValidationResult(data, None)
    except Exception as err:
        mapped = tool_error_from_exception(err)
        if mapped is None:
            raise
        return _ValidationResult({}, tool_error_envelope(*mapped))


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


def render_eval_system_prompt(runtime: EvalRuntime, tools: list[Tool[EvalRuntime]]) -> str:
    """Render the eval system prompt for a fixture snapshot and available tools."""
    inventory_section = render_home_inventory(
        runtime.snapshot,
        recorder_available=True,
        logbook_available=runtime.recorder_source.logbook_available,
    )
    return compose_system_prompt(
        runtime.settings.prompt_profile,
        runtime.case.actions_enabled,
        base_prompt=runtime.candidate.api_prompt,
        # The same available Pydantic tools provide both provider schemas and the
        # prompt summary, so candidate descriptions cannot diverge between them.
        tool_section=render_tool_capabilities(cast(list[llm.Tool], tools)),
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
    """Deterministically route requests and finish through the registered output tool."""
    user_request = _first_user_content(messages)
    tool_count = _tool_return_count(messages)
    if _last_tool_content(messages) is not None:
        last_return = _last_tool_return(messages)
        if last_return is not None:
            tool_name, content = last_return
            next_cursor = content.get("next_cursor")
            if isinstance(next_cursor, str) and tool_name in {
                TOOL_GET_HISTORY,
                TOOL_GET_STATISTICS,
                TOOL_GET_LOGBOOK,
                TOOL_GET_AUTOMATION,
            }:
                continuation_args: dict[str, object] = {"cursor": next_cursor}
                if "full living room power sensor history" in user_request.lower():
                    continuation_args["entity_ids"] = ["sensor.living_power"]
                return ModelResponse(parts=[_tool_call_part(tool_name, continuation_args, tool_count + 1)])
        followup_calls = _stub_followup_calls(user_request, tool_count)
        if followup_calls:
            return ModelResponse(
                parts=[
                    _tool_call_part(name, args, tool_count + index)
                    for index, (name, args) in enumerate(followup_calls, start=1)
                ]
            )
        return _stub_output_response(messages, info)
    return ModelResponse(
        parts=[
            _tool_call_part(name, args, index)
            for index, (name, args) in enumerate(_stub_initial_calls(user_request), start=1)
        ]
    )


def _stub_output_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Emit the same structured output-tool call that a provider must emit."""
    if not info.output_tools:
        raise RuntimeError("structured eval output tool was not registered")
    answer = EvalAnswer(answer=_all_tool_content(messages) or "", claims=_stub_claims(messages))
    output_tool = info.output_tools[0]
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name=output_tool.name,
                args=answer.model_dump(mode="json"),
                tool_call_id=f"stub-output-{_tool_return_count(messages)}",
            )
        ]
    )


def _stub_claims(messages: list[ModelMessage]) -> list[AnswerClaim]:
    """Derive a small set of claims from copied production result records."""
    claims: list[AnswerClaim] = []
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if not isinstance(part, ToolReturnPart) or not isinstance(part.content, dict):
                continue
            _claims_from_value(part.content.get("output"), claims)
            entities = part.content.get("entities")
            if isinstance(entities, dict):
                for entity_id, record in entities.items():
                    if isinstance(entity_id, str) and isinstance(record, dict):
                        _claims_from_value({"entity_id": entity_id, "rows": record.get("rows")}, claims)
            entries = part.content.get("entries")
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict) and isinstance(entry.get("entity_id"), str):
                        claims.append(  # noqa: PERF401 - constructing validated Pydantic claims is clearer here.
                            EventClaim(
                                source="logbook",
                                entity_id=entry["entity_id"],
                                event_kind="logbook_message",
                                value=str(entry.get("message", "")),
                                when=entry.get("when") if isinstance(entry.get("when"), str) else None,
                            )
                        )
    user_request = _first_user_content(messages).lower()
    history_values = [
        (claim.entity_id, float(claim.value))
        for claim in claims
        if isinstance(claim, EventClaim) and claim.source == "history" and _is_numeric(claim.value)
    ]
    if "living room temperature reading in the last 24 hours" in user_request:
        values = [value for entity_id, value in history_values if entity_id == "sensor.living_temp"]
        if values:
            claims.append(
                AggregateClaim(
                    source="history",
                    operator="maximum",
                    subject_ids=["sensor.living_temp"],
                    input_field="state",
                    input_value="state",
                    value=max(values),
                    unit="°C",
                )
            )
    if "every outside temperature reading in the last 24 hours" in user_request:
        values = [value for entity_id, value in history_values if entity_id == "sensor.tempest_temperature"]
        if values:
            claims.append(
                AggregateClaim(
                    source="history",
                    operator="maximum",
                    subject_ids=["sensor.tempest_temperature"],
                    input_field="state",
                    input_value="state",
                    value=max(values),
                    unit="°F",
                )
            )
    return claims


def _claims_from_value(value: object, claims: list[AnswerClaim]) -> None:
    if isinstance(value, list):
        for item in value:
            _claims_from_value(item, claims)
        return
    if not isinstance(value, dict):
        return
    entity_id = value.get("entity_id")
    if isinstance(entity_id, str) and "state" in value:
        claims.append(ValueClaim(subject_kind="entity", subject_id=entity_id, field="state", value=value["state"]))
    if isinstance(entity_id, str) and isinstance(value.get("duration_seconds"), int | float):
        claims.append(
            AggregateClaim(
                source="states",
                operator="duration_seconds",
                subject_ids=[entity_id],
                input_field="none",
                value=value["duration_seconds"],
                unit="seconds",
            )
        )
    attributes = value.get("attributes")
    if isinstance(entity_id, str) and isinstance(attributes, dict):
        for name, item in attributes.items():
            if isinstance(name, str) and (isinstance(item, str | int | float | bool) or item is None):
                claims.append(
                    ValueClaim(
                        subject_kind="entity",
                        subject_id=entity_id,
                        field="attribute",
                        attribute_name=name,
                        value=item,
                    )
                )
    for row in value.get("rows", ()) if isinstance(value.get("rows"), list) else ():
        if isinstance(row, list) and len(row) >= 2 and isinstance(entity_id, str):
            claims.append(  # noqa: PERF401 - constructing validated Pydantic claims is clearer here.
                EventClaim(
                    source="history",
                    entity_id=entity_id,
                    event_kind="state_transition",
                    value=str(row[1]),
                    when=row[0] if isinstance(row[0], str) else None,
                )
            )


def _tool_call_part(tool_name: str, tool_args: dict[str, object], index: int) -> ToolCallPart:
    """Build one deterministic Pydantic AI tool call part."""
    return ToolCallPart(tool_name=tool_name, args=tool_args, tool_call_id=f"stub-call-{index}")


def _stub_initial_calls(user_request: str) -> tuple[tuple[str, dict[str, object]], ...]:
    """Return the first deterministic tool call(s) for the stub model."""
    lowered = user_request.lower()
    conditional_calls = _stub_conditional_initial_calls(lowered)
    if conditional_calls is not None:
        return conditional_calls
    if "which of my evening automations controls the living room light" in lowered:
        return ((TOOL_GET_AUTOMATION, {"query": "evening living room light"}),)
    if "what does the automation called evening living room lights do" in lowered:
        return (
            (
                TOOL_GET_AUTOMATION,
                {"entity_ids": ["automation.living_scene_4f7a"], "include": ["content"]},
            ),
        )
    if "when did the evening living room lights automation most recently run" in lowered:
        return (
            (
                TOOL_GET_AUTOMATION,
                {"entity_ids": ["automation.living_scene_4f7a"], "include": ["runs"]},
            ),
        )
    if "sensor.living_room_temperature" in lowered:
        return ((TOOL_GET_HISTORY, {"entity_ids": ["sensor.living_room_temperature"], **_last_day_window()}),)
    if "full living room power sensor history" in lowered:
        return ((TOOL_GET_HISTORY, {"entity_ids": ["sensor.living_power"], **_last_day_window()}),)
    if "summarize the living room temperature history" in lowered and "humidity hourly statistics" in lowered:
        return (
            (TOOL_GET_HISTORY, {"entity_ids": ["sensor.living_temp"], **_last_day_window()}),
            (
                TOOL_GET_STATISTICS,
                {"statistic_ids": ["sensor.bedroom_humidity"], "period": "hour", **_last_day_window()},
            ),
        )
    if "find the living room temperature sensor" in lowered:
        return ((TOOL_EXECUTE_HOME_CODE, {"code": _discover_living_temperature_history_code()}),)
    if "light.living last turn on" in lowered or "light.living last turned on" in lowered:
        return (
            (
                TOOL_GET_HISTORY,
                {"entity_ids": ["light.living"], "aggregate": "last_seen", "to_state": "on", **_last_day_window()},
            ),
        )
    if "living room light turned on today" in lowered:
        return ((TOOL_GET_LOGBOOK, {"entity_ids": ["light.living"], **_today_window()}),)
    if "all lights upstairs" in lowered:
        return (
            (
                TOOL_EXECUTE_HOME_CODE,
                {
                    "code": 'await hass.services.async_call("light", "turn_off", target={"floor_id": "floor_upstairs"})\n'
                    'result = "ok"'
                },
            ),
        )
    if "living room light on right now" in lowered and "last change" in lowered:
        return ((TOOL_EXECUTE_HOME_CODE, {"code": _state_and_logbook_code()}),)
    if "light in this room" in lowered:
        return ((TOOL_GET_LOGBOOK, {"entity_ids": ["light.living"], **_today_window()}),)
    if "garage door opener" in lowered:
        return ((TOOL_EXECUTE_HOME_CODE, {"code": "result = states.entity_ids()"}),)
    if "seconds has the living room light been in its current state" in lowered:
        return ((TOOL_EXECUTE_HOME_CODE, {"code": _current_state_duration_code()}),)
    # Branch boundary: state-read of the living room temperature reads the entity so
    # the observed value surfaces in the tool return + final answer. Exclude history
    # and threshold phrasings, which are handled by their own routes or the recorder
    # fallback below.
    if "temperature" in lowered and "living room" in lowered and "history" not in lowered and "above" not in lowered:
        return ((TOOL_EXECUTE_HOME_CODE, {"code": 'result = states.get("sensor.living_temp")'}),)

    tool_name = _select_stub_tool(user_request)
    return ((tool_name, _build_stub_tool_args(tool_name, user_request)),)


def _stub_conditional_initial_calls(lowered: str) -> tuple[tuple[str, dict[str, object]], ...] | None:
    """Route the eight authored conditional requests to grounded production evidence."""
    if "living room temperature reading in the last 24 hours" in lowered and "above 25" in lowered:
        return ((TOOL_GET_HISTORY, {"entity_ids": ["sensor.living_temp"], **_last_day_window()}),)
    if "bedroom humidity is above 40%" in lowered:
        return ((TOOL_EXECUTE_HOME_CODE, {"code": _humidity_dehumidifier_code(40)}),)
    if "bedroom humidity is above 70%" in lowered:
        return ((TOOL_EXECUTE_HOME_CODE, {"code": _humidity_dehumidifier_code(70)}),)
    if "today's logbook shows the living room light turned on" in lowered:
        return ((TOOL_GET_LOGBOOK, {"entity_ids": ["light.living"], **_today_window()}),)
    if "light.living state is on after 09:00 today" in lowered:
        return ((TOOL_GET_HISTORY, {"entity_ids": ["light.living"], **_today_window()}),)
    if "living room lights group turned on after 11:00" in lowered:
        return ((TOOL_GET_LOGBOOK, {"entity_ids": ["light.living_room_lights_group"], **_today_window()}),)
    if "every outside temperature reading in the last 24 hours was below 80" in lowered:
        return ((TOOL_GET_HISTORY, {"entity_ids": ["sensor.tempest_temperature"], **_last_day_window()}),)
    if "current outside temperature is below 75" in lowered:
        return ((TOOL_EXECUTE_HOME_CODE, {"code": 'result = states.get("sensor.tempest_temperature")'}),)
    return None


def _stub_followup_calls(user_request: str, tool_count: int) -> tuple[tuple[str, dict[str, object]], ...]:
    """Return the next deterministic tool call(s) after previous tool output."""
    lowered = user_request.lower()
    if "living room temperature reading in the last 24 hours" in lowered and tool_count == 1:
        return ((TOOL_EXECUTE_HOME_CODE, {"code": _fan_50_code("fan.living_fan")}),)
    if "today's logbook shows the living room light turned on" in lowered and tool_count == 1:
        return ((TOOL_EXECUTE_HOME_CODE, {"code": _service_code("light", "turn_off", "light.living")}),)
    if "every outside temperature reading in the last 24 hours was below 80" in lowered and tool_count == 1:
        return ((TOOL_EXECUTE_HOME_CODE, {"code": _service_code("cover", "close_cover", "cover.office_blinds")}),)
    if "sensor.living_room_temperature" in lowered and tool_count == 1:
        return ((TOOL_EXECUTE_HOME_CODE, {"code": 'result = states.get("sensor.living_temp")'}),)
    if "sensor.living_room_temperature" in lowered and tool_count == 2:
        return ((TOOL_GET_HISTORY, {"entity_ids": ["sensor.living_temp"], **_last_day_window()}),)
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


def _humidity_dehumidifier_code(threshold: int) -> str:
    """Read humidity and conditionally invoke the configured dehumidifier action."""
    return (
        'state = hass.states.get("sensor.bedroom_humidity")\n'
        f"if float(state.state) > {threshold}:\n"
        '    await hass.services.async_call("switch", "toggle", target={"entity_id": "switch.dehumidifier"})\n'
        'result = {"entity_id": state.entity_id, "state": state.state}'
    )


def _fan_50_code(entity_id: str) -> str:
    """Return minimal executable code that records a fan percentage service call."""
    return (
        'await hass.services.async_call("fan", "set_percentage", {"percentage": 50}, '
        f'target={{"entity_id": "{entity_id}"}})\nresult = "ok"'
    )


def _discover_living_temperature_history_code() -> str:
    """Return one snippet that discovers the sensor before reading its history."""
    return (
        'matches = [state for state in hass.states.async_all("sensor") if state.name == "Living Temperature"]\n'
        "history = await hass.history(matches[0].entity_id, hours=24)\n"
        'result = {"entity_id": matches[0].entity_id, "history": history}'
    )


def _current_state_duration_code() -> str:
    """Calculate elapsed state duration from frozen snapshot timestamps and eval end."""
    return (
        'state = hass.states.get("light.living")\n'
        'result = {"entity_id": state.entity_id, "duration_seconds": '
        "1782734400.0 - state.last_changed_timestamp}"
    )


def _state_and_logbook_code() -> str:
    """Return one snippet that combines current state with latest logbook activity."""
    return (
        'current = hass.states.get("light.living")\n'
        'entries = await hass.logbook("light.living", hours=24)\n'
        'result = {"current": current, "entries": entries}'
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


def _last_tool_return(messages: list[ModelMessage]) -> tuple[str, dict[str, object]] | None:
    """Return the latest production tool name and mapping result."""
    for message in reversed(messages):
        if not isinstance(message, ModelRequest):
            continue
        for part in reversed(message.parts):
            if isinstance(part, ToolReturnPart) and isinstance(part.content, dict):
                return part.tool_name, part.content
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


def _is_numeric(value: object) -> bool:
    """Return whether a copied recorder scalar is numeric."""
    try:
        float(value)  # type: ignore[arg-type]
    except TypeError, ValueError:
        return False
    return True


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
