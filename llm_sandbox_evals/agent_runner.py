"""Pydantic AI agent wiring for evals over production tool cores."""

from collections.abc import Mapping
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

type _ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class _EvalTool(Protocol):
    name: str
    description: str

    def _normalize_args(self, args: Mapping[str, object]) -> dict[str, object]: ...


def build_agent(runtime: EvalRuntime, model_id: str) -> Agent[EvalRuntime, str]:
    """Build a plain-text Pydantic AI agent with production tool schemas."""
    tools = build_agent_tools(runtime)
    return Agent(
        model=make_model(model_id),
        tools=tools,
        system_prompt=render_eval_system_prompt(runtime, tools),
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
            llm_context = llm.LLMContext("test", None, "en", None, None)
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
        True,
        base_prompt=runtime.candidate.api_prompt,
        # Recorder-routing guidance is derived from the same available tools so
        # it stays consistent with the provider schemas.
        tool_section=render_tool_capabilities(cast(list[llm.Tool], tools)),
        location_section=None,
        inventory_section=inventory_section,
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
    """Invoke one stub-routable home_full action, then return plain prose."""
    _ = info
    user_request = _first_user_content(messages)
    if any(
        isinstance(part, ToolReturnPart)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
    ):
        return ModelResponse(parts=[TextPart(content="Done.")])
    action = _stub_action(user_request)
    if action is None:
        return ModelResponse(parts=[TextPart(content="Unsupported stub request.")])
    domain, service, entity_id = action
    return ModelResponse(
        parts=[
            _tool_call_part(
                TOOL_EXECUTE_HOME_CODE,
                {"code": _service_code(domain, service, entity_id)},
                1,
            )
        ]
    )


def _tool_call_part(tool_name: str, tool_args: dict[str, object], index: int) -> ToolCallPart:
    """Build one deterministic Pydantic AI tool call part."""
    return ToolCallPart(tool_name=tool_name, args=tool_args, tool_call_id=f"stub-call-{index}")


def _stub_action(user_request: str) -> tuple[str, str, str] | None:
    """Map only stub-routable home_full direct, brightness, and color requests."""
    return {
        "turn on the utility room ceiling light.": ("light", "turn_on", "light.utility_room_ceiling"),
        "turn off the utility room accent light.": ("light", "turn_off", "light.utility_room_accent"),
        "toggle the utility room outlet.": ("switch", "toggle", "switch.utility_room_outlet"),
        "set the utility room ceiling light to 50% brightness.": ("light", "turn_on", "light.utility_room_ceiling"),
        "make the utility room accent light warm white.": ("light", "turn_on", "light.utility_room_accent"),
    }.get(user_request.strip().lower())


def _service_code(domain: str, service: str, entity_id: str) -> str:
    """Return minimal executable code that records a safe service call."""
    return (
        f'await hass.services.async_call("{domain}", "{service}", target={{"entity_id": "{entity_id}"}})\n'
        'result = "ok"'
    )


def _first_user_content(messages: list[ModelMessage]) -> str:
    """Return the first user message content for deterministic stub routing."""
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    return part.content if isinstance(part.content, str) else ""
    return ""


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
