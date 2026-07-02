"""Tool runner for the dev-only eval harness."""

import json
import math
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import cast

from custom_components.llm_sandbox.const import (
    TOOL_EXECUTE_HOME_CODE,
    TOOL_GET_HISTORY,
    TOOL_GET_LOGBOOK,
    TOOL_GET_STATISTICS,
)
from custom_components.llm_sandbox.llm_api.executor import async_execute_home_code
from custom_components.llm_sandbox.llm_api.executor_support import ExecutionState
from custom_components.llm_sandbox.llm_api.facade_views import build_llm_context
from custom_components.llm_sandbox.llm_api.prompts import PromptProfile
from custom_components.llm_sandbox.llm_api.runtime import RuntimeContext
from custom_components.llm_sandbox.llm_api.tools import (
    RecoverableToolError,
    recorder_error_envelope,
    resolve_entity_ids,
)
from custom_components.llm_sandbox.runtime import SandboxSettings
from custom_components.llm_sandbox.snapshot import finalize_snapshot
from custom_components.llm_sandbox.snapshot.models import HomeSnapshot, SnapshotScope

from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.schema import EvalCase, ToolCall, ToolOutcome

EVAL_SCOPE: SnapshotScope = SnapshotScope(
    assistant="conversation",
    restrict_to_assist_exposed=False,
    exclude_hidden=True,
    excluded_entity_categories=frozenset({"config", "diagnostic"}),
)


def apply_scope(
    snapshot: HomeSnapshot,
    scope: SnapshotScope,
    *,
    anchor_device_id: str | None = None,
) -> HomeSnapshot:
    """Return a new snapshot with entities failing the offline scope checks removed.

    Mirrors production ``_passes_visibility`` for the offline-applicable fields only
    (``exclude_hidden`` + ``excluded_entity_categories``). Assist-exposure filtering
    needs live HA and stays a ``build_snapshot`` concern; the eval scope disables it.
    Collection pruning, state enrichment, and index rebuilding are delegated to the
    production snapshot finalizer.
    """
    visible: set[str] = set()
    for entity_id in snapshot.states:
        entry = snapshot.entities.get(entity_id)
        # Branch boundary: state-only entities skip registry-characteristic visibility checks.
        if entry is None:
            visible.add(entity_id)
            continue
        # Branch boundary: hidden registry entities are excluded when the eval scope asks for it.
        if scope.exclude_hidden and entry.hidden_by is not None:
            continue
        # Branch boundary: config/diagnostic registry entities are excluded by the eval scope.
        if entry.entity_category in scope.excluded_entity_categories:
            continue
        visible.add(entity_id)

    return finalize_snapshot(snapshot, visible=visible, anchor_device_id=anchor_device_id)


class RecordingInvoker:
    """Non-live service invoker: records validated ProposedAction dicts, returns None.

    This is the ONLY live seam in the executor path. It never touches Home Assistant.
    """

    def __init__(self) -> None:
        """Initialize the in-memory action recording list."""
        self.calls: list[dict[str, object]] = []

    async def __call__(self, action: dict[str, object]) -> object:
        """Record one already-validated action without dispatching to live Home Assistant."""
        # Safety constraint: copy the proposed action and never call hass.services or any live callback.
        self.calls.append(dict(action))
        return None


def _for_scoring(action: Mapping[str, object]) -> dict[str, object]:
    """Normalize a recorded action so the frozen scorer can read domain/service separately.

    Production compact records fold domain into service (``domain.service``); the
    invoker's ProposedAction already carries separate domain/service. This split
    is eval-only and never mutates the model-facing result.
    """
    normalized = dict(action)
    if "domain" not in normalized:
        service = normalized.get("service")
        if isinstance(service, str) and "." in service:
            domain, _, svc = service.partition(".")
            normalized["domain"] = domain
            normalized["service"] = svc
    return normalized


async def run_tool(
    call: ToolCall,
    case: EvalCase,
    snapshot: HomeSnapshot,
    prompt_profile: PromptProfile,
    *,
    invoker: RecordingInvoker,
) -> ToolOutcome:
    """Run a selected eval tool against the caller-provided fresh snapshot."""
    tool_name = call.tool_name
    tool_args = _tool_args(call.tool_args)

    # Branch boundary: execute_home_code uses the real production executor with a non-live invoker.
    if tool_name == TOOL_EXECUTE_HOME_CODE:
        return await _run_execute(str(tool_args.get("code", "")), case, snapshot, prompt_profile, invoker)
    # Branch boundary: recorder tools are fixture-backed and never touch a database.
    if tool_name == TOOL_GET_HISTORY:
        return _run_history(tool_args, case, snapshot)
    # Branch boundary: recorder tools are fixture-backed and never touch a database.
    if tool_name == TOOL_GET_STATISTICS:
        return _run_statistics(tool_args, case, snapshot)
    # Branch boundary: recorder tools are fixture-backed and never touch a database.
    if tool_name == TOOL_GET_LOGBOOK:
        return _run_logbook(tool_args, case, snapshot)

    # Branch boundary: unsupported model-selected tool.
    return ToolOutcome(ok=False, tool_name=tool_name, result=None, recorded_actions=(), error="unknown_tool")


async def _run_execute(
    code: str,
    case: EvalCase,
    snapshot: HomeSnapshot,
    prompt_profile: PromptProfile,
    invoker: RecordingInvoker,
) -> ToolOutcome:
    """Run execute_home_code through the production executor against a frozen snapshot."""
    area_id: str | None = None
    area_name: str | None = None
    floor_id: str | None = None
    floor_name: str | None = None

    # Branch boundary: derive request location only from the frozen snapshot device registry.
    if case.llm_context.device_id in snapshot.devices:
        device = snapshot.devices[case.llm_context.device_id]
        area_id = device.area_id
        # Branch boundary: an unassigned device has no area/floor context.
        if area_id is not None and area_id in snapshot.areas:
            area = snapshot.areas[area_id]
            area_name = area.name
            floor_id = area.floor_id
            # Branch boundary: an area may not be assigned to a floor.
            if floor_id is not None and floor_id in snapshot.floors:
                floor_name = snapshot.floors[floor_id].name

    ctx = build_llm_context(
        case.llm_context.platform,
        None,
        None,
        None,
        case.llm_context.language,
        None,
        case.llm_context.device_id,
        area_id,
        area_name,
        floor_id,
        floor_name,
    )
    settings = SandboxSettings(
        execution_timeout_seconds=10,
        helper_call_budget=20,
        scope=EVAL_SCOPE,
        actions_enabled=case.actions_enabled,
        action_domains=frozenset(),
        prompt_profile=prompt_profile,
    )
    runtime = RuntimeContext(
        state=ExecutionState(helper_call_limit=20),
        settings=settings,
        invoke=invoker,
        deadline=math.inf,
    )

    try:
        # Safety constraint: the executor receives only the frozen snapshot and RecordingInvoker live seam.
        result = await async_execute_home_code(code=code, snapshot=snapshot, llm_context=ctx, runtime=runtime)
    except Exception as err:  # noqa: BLE001
        return ToolOutcome(
            ok=False,
            tool_name=TOOL_EXECUTE_HOME_CODE,
            result=None,
            recorded_actions=tuple(_for_scoring(action) for action in invoker.calls),
            error=f"{type(err).__name__}: {err}",
        )

    result_dict = cast(dict[str, object], result)
    actions = _dict_list(result_dict.get("actions", []))
    return ToolOutcome(
        ok=True,
        tool_name=TOOL_EXECUTE_HOME_CODE,
        result=result_dict,
        recorded_actions=tuple(_for_scoring(action) for action in actions),
        error=None,
    )


def _run_history(tool_args: dict[str, object], case: EvalCase, snapshot: HomeSnapshot) -> ToolOutcome:
    """Return fixture-backed history rows in the production response envelope."""
    try:
        entity_ids = resolve_entity_ids(snapshot, tool_args, "entity_ids")
    except RecoverableToolError as err:
        return _recoverable_recorder_error(TOOL_GET_HISTORY, err, snapshot)
    start = _optional_string(tool_args.get("start"))
    end = _optional_string(tool_args.get("end"))

    fixture = get_home(case.home)
    history = _recorder_section(fixture, "history")
    entities = {entity_id: _history_entity_payload(history.get(entity_id, [])) for entity_id in entity_ids}
    return ToolOutcome(
        ok=True,
        tool_name=TOOL_GET_HISTORY,
        result={"window": {"start": start, "end": end}, "entities": entities},
        recorded_actions=(),
        error=None,
    )


def _run_statistics(tool_args: dict[str, object], case: EvalCase, snapshot: HomeSnapshot) -> ToolOutcome:
    """Return fixture-backed statistics rows in the production response envelope."""
    try:
        statistic_ids = resolve_entity_ids(snapshot, tool_args, "statistic_ids")
    except RecoverableToolError as err:
        return _recoverable_recorder_error(TOOL_GET_STATISTICS, err, snapshot)
    start = _optional_string(tool_args.get("start"))
    end = _optional_string(tool_args.get("end"))
    period = str(tool_args.get("period", "hour"))

    fixture = get_home(case.home)
    statistics = _recorder_section(fixture, "statistics")
    rows = {
        statistic_id: {"rows": [_statistics_row(row) for row in statistics.get(statistic_id, [])]}
        for statistic_id in statistic_ids
    }
    return ToolOutcome(
        ok=True,
        tool_name=TOOL_GET_STATISTICS,
        result={
            "window": {"start": start, "end": end},
            "period": period,
            "statistics": rows,
        },
        recorded_actions=(),
        error=None,
    )


def _run_logbook(tool_args: dict[str, object], case: EvalCase, snapshot: HomeSnapshot) -> ToolOutcome:
    """Return fixture-backed logbook rows in the production response envelope."""
    try:
        entity_ids = resolve_entity_ids(snapshot, tool_args, "entity_ids")
    except RecoverableToolError as err:
        return _recoverable_recorder_error(TOOL_GET_LOGBOOK, err, snapshot)
    start = _optional_string(tool_args.get("start"))
    end = _optional_string(tool_args.get("end"))

    fixture = get_home(case.home)
    logbook = _recorder_section(fixture, "logbook")
    entries = [_logbook_entry(entity_id, row) for entity_id in entity_ids for row in logbook.get(entity_id, [])]
    return ToolOutcome(
        ok=True,
        tool_name=TOOL_GET_LOGBOOK,
        result={"window": {"start": start, "end": end}, "entries": entries},
        recorded_actions=(),
        error=None,
    )


def _tool_args(value: object) -> dict[str, object]:
    """Coerce model tool args to the narrow dict shape consumed by runners."""
    # Branch boundary: malformed tool_args are treated as empty per the tool-runner contract.
    if not isinstance(value, dict):
        return {}
    return cast(dict[str, object], value)


def _optional_string(value: object) -> str | None:
    """Return an ISO string argument or None for omitted/null windows."""
    # Branch boundary: recorder windows may be omitted.
    if value is None:
        return None
    return str(value)


def _recoverable_recorder_error(tool_name: str, err: RecoverableToolError, snapshot: HomeSnapshot) -> ToolOutcome:
    """Map production recoverable recorder errors to eval response envelopes."""
    return ToolOutcome(
        ok=True,
        tool_name=tool_name,
        result=cast(dict[str, object], recorder_error_envelope(err.key, err.placeholders, snapshot)),
        recorded_actions=(),
        error=None,
    )


def tool_result_message(tool_call_id: str, result: dict[str, object] | None) -> dict[str, object]:
    """Build the provider tool-result message, bounded for replay."""
    return {"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(result)[:8000]}


def _recorder_section(fixture: ModuleType, section: str) -> dict[str, list[dict[str, object]]]:
    """Return one typed canned recorder section from a fixture module."""
    recorder = cast(Callable[[], dict[str, object]], fixture.recorder)
    data = recorder()
    return cast(dict[str, list[dict[str, object]]], data[section])


def _history_entity_payload(rows: list[dict[str, object]]) -> dict[str, object]:
    """Build the de-duplicated production history payload for one entity."""
    converted = [_history_row(row) for row in rows]
    entity: dict[str, object] = {"rows": [row for row, _unit in converted]}
    unit = next((unit for _row, unit in converted if unit), None)
    if unit is not None:
        entity["unit"] = unit
    return entity


def _history_row(row: Mapping[str, object]) -> tuple[list[object], str | None]:
    """Convert a fixture history row to ``([time, state], unit)``."""
    timestamp = row.get("last_changed") or row.get("last_updated")
    attributes = row.get("attributes")
    unit = None
    if isinstance(attributes, Mapping):
        unit = attributes.get("unit_of_measurement") or attributes.get("unit")
    return [str(timestamp), row.get("state")], str(unit) if unit is not None else None


def _statistics_row(row: Mapping[str, object]) -> list[object]:
    """Convert a fixture statistics row to the production ``[time, value]`` array."""
    timestamp = row.get("start") or row.get("end") or row.get("last_reset")
    value = row.get("state")
    if value is None:
        for key in ("mean", "sum", "min", "max"):
            value = row.get(key)
            if value is not None:
                break
    return [str(timestamp), value]


def _logbook_entry(entity_id: str, row: Mapping[str, object]) -> dict[str, object]:
    """Build one flat logbook entry with the scoped entity id retained."""
    entry = dict(row)
    entry["entity_id"] = entity_id
    return entry


def _dict_list(value: object) -> list[dict[str, object]]:
    """Coerce a JSON-like list of dicts into the ToolOutcome action tuple shape."""
    # Branch boundary: unexpected executor action payloads are ignored rather than re-shaped unsafely.
    if not isinstance(value, list):
        return []
    return [cast(dict[str, object], item) for item in value if isinstance(item, dict)]
