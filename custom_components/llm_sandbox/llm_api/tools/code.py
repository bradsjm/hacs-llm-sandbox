"""The ``execute_home_code`` LLM tool.

Runs bounded Python/Monty code against a frozen, read-only Home Assistant
view. The tool wires the schema and dispatch; the actual Monty run lives in
``executor.py``.
"""

import logging
import time
from typing import Any, cast, final, override

import voluptuous as vol
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import llm
from homeassistant.util.json import JsonObjectType

from ...const import TOOL_EXECUTE_HOME_CODE
from ...snapshot import build_snapshot
from ...snapshot.models import HomeSnapshot
from ...types import ProposedAction
from ..errors import setup_error_payload, tool_error_from_exception
from ..executor import MAX_MONTY_CODE_CHARS, async_execute_home_code
from ..executor_support import ExecutionState
from ..facade_views import build_llm_context
from ..prompts import build_execute_home_code_description
from ..runtime import RuntimeContext
from ._support import _require_loaded_entry, _require_loaded_entry_error

_LOGGER = logging.getLogger(__name__)

type ToolArgs = dict[str, object]


@final
class ExecuteHomeCodeTool(llm.Tool):
    """Run bounded Home Assistant code in the Monty sandbox."""

    name = TOOL_EXECUTE_HOME_CODE
    description = build_execute_home_code_description()
    parameters: vol.Schema = vol.Schema(
        {
            vol.Required(
                "code",
                description="Python-like Monty code. Assign final JSON-safe data to result.",
            ): vol.All(str, vol.Length(min=1, max=MAX_MONTY_CODE_CHARS)),
        }
    )

    def __init__(self, entry_id: str) -> None:
        """Initialize the code execution tool for one config entry."""
        self.entry_id = entry_id

    @override
    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        try:
            data = cast(ToolArgs, self.parameters(tool_input.tool_args))
            return await _execute(hass, self.entry_id, data, llm_context)
        except Exception as err:
            mapped = tool_error_from_exception(err)
            if mapped is None:
                raise
            key, placeholders = mapped
            return cast(JsonObjectType, setup_error_payload(key, placeholders))


async def _execute(
    hass: HomeAssistant,
    entry_id: str,
    data: ToolArgs,
    llm_context: llm.LLMContext,
) -> JsonObjectType:
    """Build a Monty view, construct facades, and run the executor."""
    setup_error = _require_loaded_entry_error(hass, entry_id)
    if setup_error is not None:
        key, placeholders = setup_error
        return cast(JsonObjectType, setup_error_payload(key, placeholders))
    typed_entry = _require_loaded_entry(hass, entry_id)

    runtime_data = typed_entry.runtime_data
    assert runtime_data is not None
    settings = runtime_data.settings
    code = cast(str, data["code"])
    deadline = time.monotonic() + settings.execution_timeout_seconds

    # Build a fresh Monty view on the event loop before execution.
    snapshot = build_snapshot(
        hass,
        scope=settings.scope,
        anchor_device_id=llm_context.device_id,
    )

    # Build the LLM context view from the live request metadata.
    real_context = llm_context.context if llm_context.context is not None else Context()
    area_id, area_name, floor_id, floor_name = _snapshot_location(snapshot, llm_context.device_id)
    safe_context = build_llm_context(
        platform=llm_context.platform,
        context_id=real_context.id,
        parent_id=real_context.parent_id,
        user_id=real_context.user_id,
        language=llm_context.language,
        assistant=llm_context.assistant,
        device_id=llm_context.device_id,
        area_id=area_id,
        area_name=area_name,
        floor_id=floor_id,
        floor_name=floor_name,
    )

    async def _invoke(action: ProposedAction) -> object:
        return await hass.services.async_call(
            cast(str, action["domain"]),
            cast(str, action["service"]),
            service_data=cast(dict[str, Any] | None, action["service_data"] or None),
            target=cast(dict[str, Any] | None, action["target"]),
            blocking=cast(bool, action["blocking"]),
            return_response=cast(bool, action["return_response"]),
            context=real_context,
        )

    runtime = RuntimeContext(
        state=ExecutionState(helper_call_limit=settings.helper_call_budget),
        settings=settings,
        invoke=_invoke,
        deadline=deadline,
    )

    result = await async_execute_home_code(
        code,
        snapshot=snapshot,
        llm_context=safe_context,
        runtime=runtime,
    )
    if isinstance(result, dict):
        execution = result.get("execution", {})
        _LOGGER.debug(
            "execute_home_code: status=%s helper_calls=%s/%s actions=%d",
            execution.get("status") if isinstance(execution, dict) else "n/a",
            runtime.state.helper_calls,
            runtime.state.helper_call_limit,
            len(runtime.state.actions),
        )
    return cast(JsonObjectType, result)


def _snapshot_location(
    snapshot: HomeSnapshot,
    device_id: str | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Resolve initiating-device location from the fresh Monty view."""
    if device_id is None:
        return None, None, None, None

    device = snapshot.devices.get(device_id)
    area_id = device.area_id if device is not None else None
    area = snapshot.areas.get(area_id) if area_id is not None else None
    floor_id = area.floor_id if area is not None else None
    floor = snapshot.floors.get(floor_id) if floor_id is not None else None
    return (
        area_id,
        area.name if area is not None else None,
        floor_id,
        floor.name if floor is not None else None,
    )
