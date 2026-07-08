"""The ``execute_home_code`` LLM tool.

Runs bounded Python/Monty code against a frozen, read-only Home Assistant
view. The tool wires the schema and dispatch; the actual Monty run lives in
``executor.py``.
"""

import logging
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any, cast, final, override

import voluptuous as vol
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import llm
from homeassistant.util.json import JsonObjectType

from ...const import TOOL_EXECUTE_HOME_CODE
from ...runtime import SandboxRuntime
from ...snapshot import build_snapshot
from ...snapshot.models import HomeSnapshot
from ...types import ProposedAction
from ..errors import RecoverableToolError, setup_error_payload, tool_error_from_exception
from ..executor import MAX_MONTY_CODE_CHARS, async_execute_home_code
from ..executor_support import ExecutionState
from ..facade_views import build_llm_context
from ..prompts import build_execute_home_code_description
from ..runtime import RuntimeContext
from ._support import _require_loaded_entry_error, _require_sandbox_runtime
from .recorder import RECORDER_UNAVAILABLE, fetch_flat_history_rows, fetch_flat_statistics_rows, recorder_available

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
        # Schema validation FIRST: eval validates equivalently before calling
        # run_execute, so both surfaces see identical invalid_tool_input errors.
        try:
            data = cast(ToolArgs, self.parameters(tool_input.tool_args))
        except Exception as err:
            mapped = tool_error_from_exception(err)
            if mapped is None:
                raise
            key, placeholders = mapped
            return cast(JsonObjectType, setup_error_payload(key, placeholders))

        setup_error = _require_loaded_entry_error(hass, self.entry_id)
        if setup_error is not None:
            key, placeholders = setup_error
            return cast(JsonObjectType, setup_error_payload(key, placeholders))
        sandbox_runtime = _require_sandbox_runtime(hass, self.entry_id)
        settings = sandbox_runtime.settings
        deadline = time.monotonic() + settings.execution_timeout_seconds

        # Build a fresh Monty view on the event loop before execution.
        snapshot = build_snapshot(
            hass,
            scope=settings.scope,
            anchor_device_id=llm_context.device_id,
        )
        runtime = _production_runtime(hass, snapshot, llm_context, sandbox_runtime, deadline)
        return await self.run_execute(snapshot, data, llm_context, runtime)

    async def run_execute(
        self,
        snapshot: HomeSnapshot,
        data: ToolArgs,
        llm_context: llm.LLMContext,
        runtime: RuntimeContext,
    ) -> JsonObjectType:
        """Run execute_home_code and envelope recoverable failures.

        Hass-free public entry: ``data`` is already schema-validated, ``snapshot``
        is already built, and ``runtime`` supplies host-only seams. Error mapping
        lives here so eval calls see byte-identical output to live calls.
        """
        try:
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

            result = await async_execute_home_code(
                cast(str, data["code"]),
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
        except Exception as err:
            mapped = tool_error_from_exception(err)
            if mapped is None:
                raise
            key, placeholders = mapped
            return cast(JsonObjectType, setup_error_payload(key, placeholders))


def _production_runtime(
    hass: HomeAssistant,
    snapshot: HomeSnapshot,
    llm_context: llm.LLMContext,
    sandbox_runtime: SandboxRuntime,
    deadline: float,
) -> RuntimeContext:
    """Build a RuntimeContext backed by live Home Assistant host seams."""
    settings = sandbox_runtime.settings
    real_context = llm_context.context if llm_context.context is not None else Context()

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

    async def _fetch_history(entity_ids: Sequence[str], start: datetime, end: datetime) -> list[dict[str, object]]:
        # Private host-side recorder seam: validates against the fresh snapshot
        # and never passes live hass/recorder objects into Monty-visible inputs.
        # Sync only when this run dispatched a live write (read-after-write);
        # standalone tools keep the unconditional default sync=True.
        if not recorder_available(hass):
            raise RecoverableToolError(RECORDER_UNAVAILABLE, {})
        return await fetch_flat_history_rows(
            hass, snapshot, deadline, list(entity_ids), start, end, sync=state.live_write_dispatched
        )

    async def _fetch_statistics(entity_ids: Sequence[str], start: datetime, end: datetime) -> list[dict[str, object]]:
        # Private host-side statistics seam; statistics are recorder-derived live
        # reads but cross into Monty only as JSON-safe rows.
        if not recorder_available(hass):
            raise RecoverableToolError(RECORDER_UNAVAILABLE, {})
        return await fetch_flat_statistics_rows(
            hass, snapshot, deadline, list(entity_ids), start, end, sync=state.live_write_dispatched
        )

    async def _run_blocking(fn: Callable[[], object]) -> object:
        return await hass.async_add_executor_job(fn)

    # Hoist state so the fetcher closures can read the live-write flag.
    state = ExecutionState(helper_call_limit=settings.helper_call_budget)
    return RuntimeContext(
        state=state,
        settings=settings,
        invoke=_invoke,
        fetch_history=_fetch_history,
        fetch_statistics=_fetch_statistics,
        run_blocking=_run_blocking,
        deadline=deadline,
        memory=sandbox_runtime.memory_store.for_context(llm_context),
    )


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
