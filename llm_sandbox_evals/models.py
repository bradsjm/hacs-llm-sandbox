"""Model adapters for native tool-calling eval turns."""

import asyncio
import importlib
import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import Protocol, cast

from custom_components.llm_sandbox.const import (
    TOOL_EXECUTE_HOME_CODE,
    TOOL_GET_HISTORY,
    TOOL_GET_LOGBOOK,
    TOOL_GET_STATISTICS,
)

from llm_sandbox_evals.schema import AgentStep, ToolCall

_ENTITY_ID_RE = re.compile(r"\b[a-z_]+\.[a-z0-9_]+\b")
_FIXED_END = "2026-06-29T12:00:00+00:00"
_FIXED_START = "2026-06-28T12:00:00+00:00"
_FIXED_TODAY_START = "2026-06-29T00:00:00+00:00"
_LITELLM_TIMEOUT_BUFFER = 5.0


def litellm_reasoning_kwargs(*, temperature: float, reasoning_effort: str | None) -> dict[str, object]:
    """Map decoding intent onto litellm kwargs honoring the reasoning contract."""
    # Branch boundary: no reasoning requested -> deterministic temperature only.
    if reasoning_effort is None:
        return {"temperature": temperature}
    kwargs: dict[str, object] = {"extra_body": {"reasoning_effort": reasoning_effort}}
    # Branch boundary: explicit no-reasoning keeps deterministic decoding.
    if reasoning_effort == "none":
        kwargs["temperature"] = temperature
    return kwargs


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


class LiteLLMAdapter:
    """Adapter for real models routed through LiteLLM."""

    def __init__(self, reasoning_effort: str | None = None, *, model_timeout: float = 75.0) -> None:
        """Store model-call options forwarded to LiteLLM."""
        self._reasoning_effort = reasoning_effort
        self._model_timeout = model_timeout

    async def respond(
        self,
        model_id: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> AgentStep:
        """Call LiteLLM and normalize one assistant message into an AgentStep."""
        try:
            litellm = importlib.import_module("litellm")
            _quiet_litellm(litellm)
            acompletion = cast(Callable[..., Awaitable[object]], litellm.__dict__["acompletion"])
            start = time.monotonic()
            litellm_timeout = max(1.0, self._model_timeout - _LITELLM_TIMEOUT_BUFFER)
            kwargs: dict[str, object] = {
                "model": model_id,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "timeout": litellm_timeout,
                "num_retries": 0,
            }
            kwargs.update(litellm_reasoning_kwargs(temperature=0.0, reasoning_effort=self._reasoning_effort))
            response = await asyncio.wait_for(acompletion(**kwargs), timeout=self._model_timeout)
            return _step_from_litellm_response(response)
        except TimeoutError as err:
            raise ModelResponseError(
                f"LiteLLM completion timed out after {self._model_timeout:g} seconds",
                detail=_timeout_error_detail(
                    model_id=model_id,
                    model_timeout=self._model_timeout,
                    litellm_timeout=litellm_timeout,
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
    return LiteLLMAdapter(reasoning_effort=reasoning_effort, model_timeout=model_timeout)


def _step_from_litellm_response(response: object) -> AgentStep:
    """Extract ``choices[0].message`` from a LiteLLM response object."""
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise ValueError("LiteLLM response did not include choices")

    message = getattr(choices[0], "message", None)
    content = _message_value(message, "content")
    text = content if isinstance(content, str) else ""
    raw_tool_calls = _message_value(message, "tool_calls")
    tool_calls = _parse_tool_calls(raw_tool_calls)
    assistant_message: dict[str, object] = {"role": "assistant", "content": text}
    # Branch boundary: replay tool_calls verbatim enough for provider continuation.
    if isinstance(raw_tool_calls, list):
        assistant_message["tool_calls"] = [_plain_tool_call(item) for item in raw_tool_calls]
    return AgentStep(
        tool_calls=tool_calls, text=text, assistant_message=assistant_message, raw=json.dumps(assistant_message)
    )


def _quiet_litellm(litellm: object) -> None:
    """Disable LiteLLM debug chatter so eval stderr is owned by the harness."""
    for name, value in (("suppress_debug_info", True), ("set_verbose", False)):
        if hasattr(litellm, name):
            setattr(litellm, name, value)


def _provider_error_detail(err: BaseException, *, model_id: str) -> str:
    """Return useful LiteLLM/provider diagnostics without dumping request payloads."""
    lines = [f"requested model: {model_id}", _exception_line(err)]
    metadata = _exception_metadata(err)
    if metadata:
        lines.append("provider metadata: " + ", ".join(metadata))

    response = getattr(err, "response", None)
    response_status = getattr(response, "status_code", None)
    response_text = getattr(response, "text", None)
    if response_status is not None:
        lines.append(f"response status: {response_status}")
    if isinstance(response_text, str) and response_text:
        lines.append("response body: " + _limit_detail(response_text))

    cause = err.__cause__ or err.__context__
    while cause is not None:
        lines.append("caused by: " + _exception_line(cause))
        cause = cause.__cause__ or cause.__context__
    return "\n".join(lines)


def _timeout_error_detail(*, model_id: str, model_timeout: float, litellm_timeout: float, elapsed: float) -> str:
    """Return an actionable diagnostic when the provider never returns a response."""
    return "\n".join(
        (
            f"requested model: {model_id}",
            f"model generation exceeded the eval timeout after {elapsed:.1f}s",
            f"eval model timeout: {model_timeout:g}s",
            f"per-request timeout sent to LiteLLM: {litellm_timeout:g}s",
            "OpenRouter did not return an HTTP status or response body before this timeout",
            "OpenRouter's global status page can still be healthy when an individual free-model generation queues or runs this long",
            "increase --model-timeout for slow/free models, lower --concurrency, or use the paid/non-free model id if you need reliable eval throughput",
        )
    )


def _exception_line(err: BaseException) -> str:
    """Format one exception with its fully qualified type."""
    return f"{type(err).__module__}.{type(err).__name__}: {err}"


def _exception_metadata(err: BaseException) -> list[str]:
    """Extract common LiteLLM provider metadata from exception attributes."""
    metadata: list[str] = []
    for name in ("status_code", "llm_provider", "model", "code", "param"):
        value = getattr(err, name, None)
        if value is not None:
            metadata.append(f"{name}={value!r}")
    return metadata


def _limit_detail(value: str) -> str:
    """Bound provider response details so one failing model does not flood stderr."""
    compact = " ".join(value.split())
    if len(compact) <= 1000:
        return compact
    return compact[:1000] + "..."


def _parse_tool_calls(raw_tool_calls: object) -> tuple[ToolCall, ...]:
    """Parse provider tool calls into the eval harness contract."""
    if not isinstance(raw_tool_calls, list):
        return ()
    parsed: list[ToolCall] = []
    for index, item in enumerate(raw_tool_calls):
        function = _message_value(item, "function")
        name = _message_value(function, "name")
        arguments = _message_value(function, "arguments")
        if not isinstance(name, str):
            continue
        try:
            decoded = json.loads(arguments) if isinstance(arguments, str) and arguments else {}
        except json.JSONDecodeError:
            decoded = {}
        tool_args = dict(decoded) if isinstance(decoded, dict) else {}
        call_id = _message_value(item, "id")
        parsed.append(ToolCall(id=str(call_id or f"call-{index}"), tool_name=name, tool_args=tool_args))
    return tuple(parsed)


def _plain_tool_call(item: object) -> dict[str, object]:
    """Convert one LiteLLM tool-call object to a replayable plain dict."""
    function = _message_value(item, "function")
    return {
        "id": str(_message_value(item, "id") or ""),
        "type": str(_message_value(item, "type") or "function"),
        "function": {
            "name": str(_message_value(function, "name") or ""),
            "arguments": str(_message_value(function, "arguments") or "{}"),
        },
    }


def _message_value(message: object, key: str) -> object:
    """Read a key from either provider dicts or LiteLLM message objects."""
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


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
