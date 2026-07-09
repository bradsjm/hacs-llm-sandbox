"""Recorder-backed read-only LLM tools."""

import time
from collections.abc import Mapping
from datetime import datetime
from typing import Literal, cast, final, override

import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import llm
from homeassistant.util import dt as dt_util
from homeassistant.util.json import JsonObjectType

from ...const import (
    DEFAULT_HISTORY_WINDOW_HOURS,
    DEFAULT_LOGBOOK_WINDOW_HOURS,
    DEFAULT_STATISTICS_WINDOW_HOURS,
    MAX_HISTORY_ATTRIBUTES,
    MAX_HISTORY_STATES,
    MAX_LOGBOOK_ENTRIES,
    MAX_RECORDER_ENTITY_IDS,
    MAX_RECORDER_LOOKBACK_HOURS,
    MAX_STATISTICS_LOOKBACK_HOURS,
    MAX_STATISTICS_ROWS,
    TOOL_GET_HISTORY,
    TOOL_GET_LOGBOOK,
    TOOL_GET_STATISTICS,
)
from ...snapshot import build_recorder_snapshot
from ...snapshot.models import HomeSnapshot
from ...types import TranslationPlaceholders
from ..data.history import AGGREGATORS, AggregateMode
from ..data.recorder_scope import (
    ENTITY_NOT_VISIBLE,
    SELECTOR_NO_MATCH,
    resolve_entity_ids,
)
from ..errors import RecoverableToolError, tool_error_envelope, tool_error_from_exception
from ..guidance import FailureContext, Intent, advise
from ..prompts import build_get_history_description, build_get_logbook_description, build_get_statistics_description
from ._cursor import _LOGBOOK_CURSOR_KEY, INVALID_CURSOR, Cursor, paginate_stream
from ._recorder_runtime import (
    _ALL_STAT_QUERY_TYPES,
    STATISTIC_VALUE_TYPES,
    RecorderSource,
    StatisticQueryType,
    StatisticValueType,
    _aggregate_history,
    _declarative_history,
    _logbook_when,
    _paginate_streams,
    _production_recorder_source,
    _resolve_window,
    _state_row_to_dict,
    _statistic_fields,
    _statistic_row_to_dict,
    _windowed_payload,
    recorder_available,
)
from ._recorder_runtime import (
    _sync_recorder_for_query as _sync_recorder_for_query,
)
from ._recorder_runtime import (
    fetch_flat_history_rows as fetch_flat_history_rows,
)
from ._recorder_runtime import (
    fetch_flat_short_term_statistics_rows as fetch_flat_short_term_statistics_rows,
)
from ._recorder_runtime import (
    fetch_flat_statistics_rows as fetch_flat_statistics_rows,
)
from ._recorder_runtime import (
    fetch_visible_history_rows as fetch_visible_history_rows,
)
from ._recorder_runtime import (
    logbook_available as logbook_available,
)
from ._support import _omit_empty_optional_args, _require_loaded_entry_error, _require_sandbox_runtime

RECORDER_UNAVAILABLE = "recorder_unavailable"
QUERY_FAILED = "query_failed"

# Optional recorder keys whose null value is dropped before schema validation
# (Postel's law): every optional argument the recorder tools accept.
_RECORDER_NULL_OMIT: frozenset[str] = frozenset(
    {
        "start",
        "end",
        "area_id",
        "device_id",
        "floor_id",
        "label_id",
        "domain",
        "from_state",
        "to_state",
        "bucket",
        "order_by",
        "cursor",
        "entity_ids",
        "statistic_ids",
        "attributes",
        "group_by",
        "where",
        "types",
        "hours",
        "aggregate",
        "limit",
        "period",
    }
)
# Optional scalar keys whose empty-string value is dropped before validation.
_RECORDER_EMPTY_STRING_OMIT: frozenset[str] = frozenset(
    {
        "start",
        "end",
        "area_id",
        "device_id",
        "floor_id",
        "label_id",
        "domain",
        "from_state",
        "to_state",
        "bucket",
        "order_by",
        "cursor",
    }
)
# Optional list keys whose empty-list value is dropped before validation.
_RECORDER_EMPTY_LIST_OMIT: frozenset[str] = frozenset(
    {"entity_ids", "statistic_ids", "attributes", "group_by", "where", "types"}
)
# Relative window size in hours, accepted by every recorder tool as an
# alternative to absolute ISO start/end (the sandbox forbids timedelta math).
_HOURS_ARG = vol.All(vol.Coerce(float), vol.Range(min=0))


def recorder_error_envelope(
    key: str,
    placeholders: TranslationPlaceholders,
    snapshot: HomeSnapshot | None = None,
) -> JsonObjectType:
    """Build a recoverable recorder error envelope with actionable guidance."""
    if key == ENTITY_NOT_VISIBLE:
        entity_id = placeholders.get("entity_id", "the requested entity")
        guidance = None
        # Entity visibility failures have a concrete requested entity to recover.
        if snapshot is not None:
            guidance = advise(
                snapshot,
                FailureContext(
                    intent=Intent.QUERY_HISTORY,
                    requested=entity_id,
                    domain=entity_id.split(".", 1)[0],
                ),
            ).to_payload()
        return tool_error_envelope(
            key,
            placeholders,
            message=f"Entity '{entity_id}' is not visible to this LLM tool.",
            guidance=guidance,
        )
    if key == SELECTOR_NO_MATCH:
        selectors = placeholders.get("selectors", "")
        requested = placeholders.get("selector_id", selectors or "requested selector")
        guidance = None
        # Selector failures recover from the concrete selector id when available.
        if snapshot is not None:
            guidance = advise(
                snapshot,
                FailureContext(
                    intent=Intent.RESOLVE_SELECTOR,
                    requested=requested,
                    domain=placeholders.get("domain", ""),
                ),
            ).to_payload()
        return tool_error_envelope(
            key,
            placeholders,
            message=f"Selector(s) {selectors or 'requested'} matched no visible entities.",
            guidance=guidance,
        )
    return tool_error_envelope(key, placeholders)


def _iso_datetime(value: object) -> datetime:
    """Validate an ISO datetime string or datetime object and return UTC-aware datetime."""
    if isinstance(value, datetime):
        return dt_util.as_utc(value)
    if isinstance(value, str):
        parsed = dt_util.parse_datetime(value)
        if parsed is not None:
            return dt_util.as_utc(parsed)
    raise vol.Invalid("expected an ISO datetime")


class _RecorderTool(llm.Tool):
    """Shared base for recorder-backed read-only tools."""

    name: str
    description: str
    parameters: vol.Schema

    def __init__(self, entry_id: str) -> None:
        """Initialize the recorder tool for one config entry."""
        self.entry_id = entry_id

    @override
    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        # Schema validation FIRST: the eval path validates equivalently before
        # calling run_query, so both surfaces see identical invalid_tool_input errors.
        try:
            data = cast(dict[str, object], self.parameters(self._normalize_args(tool_input.tool_args)))
        except Exception as err:
            mapped = tool_error_from_exception(err)
            if mapped is None:
                raise
            return recorder_error_envelope(*mapped)

        if not recorder_available(hass):
            return recorder_error_envelope(RECORDER_UNAVAILABLE, {})

        setup_error = _require_loaded_entry_error(hass, self.entry_id)
        if setup_error is not None:
            key, placeholders = setup_error
            return recorder_error_envelope(key, placeholders)
        settings = _require_sandbox_runtime(hass, self.entry_id).settings
        # Build a fresh visible snapshot for every recorder tool call.
        snapshot = build_recorder_snapshot(
            hass,
            scope=settings.scope,
            anchor_device_id=llm_context.device_id,
        )
        deadline = time.monotonic() + settings.execution_timeout_seconds
        source = _production_recorder_source(hass, snapshot, deadline)
        return await self.run_query(snapshot, data, source)

    async def run_query(
        self,
        snapshot: HomeSnapshot,
        data: dict[str, object],
        source: RecorderSource,
    ) -> JsonObjectType:
        """Run the concrete recorder query and envelope recoverable failures.

        Hass-free public entry: ``data`` is already schema-validated, ``source``
        provides the window anchor and async row fetchers. Recoverable failures and
        unexpected query errors are converted to LLM-visible envelopes here so the
        eval path (which calls this directly) sees byte-identical output to live.
        """
        try:
            return await self._query(snapshot, data, source)
        except RecoverableToolError as err:
            return recorder_error_envelope(err.key, err.placeholders, snapshot)
        except Exception as err:  # noqa: BLE001 - recorder tools map unexpected query failures to envelopes
            mapped = tool_error_from_exception(err)
            if mapped is None:
                return recorder_error_envelope(QUERY_FAILED, {"error": type(err).__name__})
            return recorder_error_envelope(*mapped)

    def _normalize_args(self, args: Mapping[str, object]) -> dict[str, object]:
        """Normalize tool-specific input aliases before voluptuous validation."""
        # Drop empty/null optional values so they behave as if omitted (Postel's
        # law) before the schema validates them.
        return _omit_empty_optional_args(
            args,
            null_keys=_RECORDER_NULL_OMIT,
            empty_string_keys=_RECORDER_EMPTY_STRING_OMIT,
            empty_list_keys=_RECORDER_EMPTY_LIST_OMIT,
        )

    async def _query(
        self,
        snapshot: HomeSnapshot,
        data: dict[str, object],
        source: RecorderSource,
    ) -> JsonObjectType:
        """Run the concrete recorder query body; raise RecoverableToolError on failure."""
        raise NotImplementedError


# HA-native target selectors accepted as an alternative to enumerated IDs.
RECORDER_SELECTOR_FIELD_NAMES = ("area_id", "device_id", "floor_id", "label_id", "domain")
_SELECTOR_FIELD_DESCRIPTIONS: dict[str, str] = {
    "area_id": "Area ID(s) to scope the query.",
    "device_id": "Device ID(s) to scope the query.",
    "floor_id": "Floor ID(s) to scope the query.",
    "label_id": "Label ID(s) to scope the query.",
    "domain": "Domain(s) (e.g. light) to keep from the resolved set.",
}
_SELECTOR_FIELDS: dict[vol.Optional, object] = {
    vol.Optional(field_name, description=_SELECTOR_FIELD_DESCRIPTIONS[field_name]): vol.All(cv.ensure_list, [str])
    for field_name in RECORDER_SELECTOR_FIELD_NAMES
}


@final
class GetHistoryTool(_RecorderTool):
    """Return recorded state history for visible entities."""

    name = TOOL_GET_HISTORY
    description = build_get_history_description()
    parameters: vol.Schema = vol.Schema(
        {
            vol.Optional("entity_ids", description="One or up to 20 entity IDs."): vol.All(
                cv.ensure_list,
                [cv.entity_id],
                vol.Length(min=1, max=MAX_RECORDER_ENTITY_IDS),
            ),
            **_SELECTOR_FIELDS,
            vol.Optional(
                "hours", description="Relative window size in hours; used when start/end are omitted."
            ): _HOURS_ARG,
            vol.Optional("start", description="Window start (ISO-8601). Default now-1h."): _iso_datetime,
            vol.Optional("end", description="Window end (ISO-8601). Default now."): _iso_datetime,
            vol.Optional(
                "attributes",
                description=(
                    "Optional attribute names to include per row. Each row then appends "
                    "{name: value} for requested attributes present on that row; absent names are omitted."
                ),
            ): vol.All(cv.ensure_list, [str], vol.Length(min=1, max=MAX_HISTORY_ATTRIBUTES)),
            vol.Optional(
                "aggregate",
                description="Optional server-side summary mode instead of raw rows.",
            ): vol.Any(vol.In(tuple(AGGREGATORS)), dict),
            vol.Optional("group_by", description="Optional analytics group key(s)."): vol.All(
                cv.ensure_list, [str], vol.Length(min=1, max=4)
            ),
            vol.Optional("bucket", description="Optional analytics bucket, e.g. 15m, 1h, 1d."): str,
            vol.Optional("where", description="Optional analytics row filters."): vol.All(cv.ensure_list, [dict]),
            vol.Optional("order_by", description="Optional analytics result sort field; prefix '-' for desc."): str,
            vol.Optional("limit", description="Maximum analytics result rows."): vol.All(
                vol.Coerce(int), vol.Range(min=1)
            ),
            vol.Optional(
                "from_state",
                description="Optional count_transitions filter for the previous state.",
            ): str,
            vol.Optional(
                "to_state",
                description="Optional count_transitions filter for the next state.",
            ): str,
            vol.Optional(
                "cursor",
                description=(
                    "Opaque cursor from this tool's prior next_cursor for the same resolved scope; "
                    "omit start, end, and hours when using it."
                ),
            ): str,
        }
    )

    def _normalize_args(self, args: Mapping[str, object]) -> dict[str, object]:
        """Accept the requested history analytics input-key synonyms."""
        data = dict(args)
        if "agg" in data and "aggregate" not in data:
            data["aggregate"] = data.pop("agg")
        if "groupby" in data and "group_by" not in data:
            data["group_by"] = data.pop("groupby")
        if "resample" in data and "bucket" not in data:
            data["bucket"] = data.pop("resample")
        if "interval" in data and "bucket" not in data:
            data["bucket"] = data.pop("interval")
        # Aliases are canonicalized first so their empty/null values drop
        # through the shared omission (e.g. resample:"" -> bucket:"" -> omitted).
        return _omit_empty_optional_args(
            data,
            null_keys=_RECORDER_NULL_OMIT,
            empty_string_keys=_RECORDER_EMPTY_STRING_OMIT,
            empty_list_keys=_RECORDER_EMPTY_LIST_OMIT,
        )

    @override
    async def _query(
        self,
        snapshot: HomeSnapshot,
        data: dict[str, object],
        source: RecorderSource,
    ) -> JsonObjectType:
        entity_ids = resolve_entity_ids(snapshot, data, "entity_ids")
        requested_attributes = cast(list[str] | None, data.get("attributes"))
        aggregate = cast(AggregateMode | None, data.get("aggregate"))
        analytics_requested = any(
            key in data for key in ("aggregate", "group_by", "bucket", "where", "order_by", "limit")
        )
        if isinstance(aggregate, str) and not any(
            key in data for key in ("group_by", "bucket", "where", "order_by", "limit")
        ):
            return await _aggregate_history(source, entity_ids, data, aggregate)
        if analytics_requested:
            return await _declarative_history(snapshot, source, entity_ids, data)

        start, end, cursor = _resolve_window(
            data,
            now=source.now,
            default_hours=DEFAULT_HISTORY_WINDOW_HOURS,
            max_hours=MAX_RECORDER_LOOKBACK_HOURS,
            expected_kind="history",
            expected_scope_ids=tuple(sorted(entity_ids)),
        )
        result = await source.fetch_history(entity_ids, start, end)
        budget = max(1, MAX_HISTORY_STATES // len(entity_ids))
        stream_rows: dict[str, list[list[object]]] = {}
        stream_units: dict[str, str] = {}
        for entity_id, states in result.items():
            converted = [_state_row_to_dict(row, requested_attributes) for row in states]
            stream_rows[entity_id] = [row for row, _unit in converted]
            unit = next((unit for _row, unit in converted if unit), None)
            if unit:
                stream_units[entity_id] = unit
        pages, next_cutoffs = _paginate_streams(stream_rows, budget=budget, cutoffs=cursor.cutoffs)
        entities: dict[str, dict[str, object]] = {}
        for entity_id, page in pages.items():
            entity: dict[str, object] = {"rows": page}
            if (unit := stream_units.get(entity_id)) is not None:
                entity["unit"] = unit
            entities[entity_id] = entity

        return _windowed_payload(
            start,
            end,
            {"entities": entities},
            Cursor(kind="history", scope_ids=cursor.scope_ids, start=start, end=end, cutoffs=next_cutoffs)
            if next_cutoffs
            else None,
        )


@final
class GetStatisticsTool(_RecorderTool):
    """Return long-term recorder statistics for visible statistic IDs."""

    name = TOOL_GET_STATISTICS
    description = build_get_statistics_description()
    parameters: vol.Schema = vol.Schema(
        {
            vol.Optional(
                "statistic_ids",
                description="One or up to 20 statistic IDs (usually entity IDs).",
            ): vol.All(
                cv.ensure_list,
                [str],
                vol.Length(min=1, max=MAX_RECORDER_ENTITY_IDS),
            ),
            **_SELECTOR_FIELDS,
            vol.Optional(
                "hours", description="Relative window size in hours; used when start/end are omitted."
            ): _HOURS_ARG,
            vol.Optional("start", description="Window start (ISO-8601). Default now-24h."): _iso_datetime,
            vol.Optional("end", description="Window end (ISO-8601). Default now."): _iso_datetime,
            vol.Optional("period", default="hour", description="Aggregation bucket."): vol.In(
                ("5minute", "hour", "day")
            ),
            vol.Optional(
                "types",
                description="Optional statistic value fields to include per row.",
            ): vol.All(
                cv.ensure_list,
                [vol.In(STATISTIC_VALUE_TYPES)],
                vol.Length(min=1, max=len(STATISTIC_VALUE_TYPES)),
            ),
            vol.Optional(
                "cursor",
                description=(
                    "Opaque cursor from this tool's prior next_cursor for the same resolved scope; "
                    "omit start, end, and hours when using it."
                ),
            ): str,
        }
    )

    @override
    async def _query(
        self,
        snapshot: HomeSnapshot,
        data: dict[str, object],
        source: RecorderSource,
    ) -> JsonObjectType:
        statistic_ids = resolve_entity_ids(snapshot, data, "statistic_ids")
        start, end, cursor = _resolve_window(
            data,
            now=source.now,
            default_hours=DEFAULT_STATISTICS_WINDOW_HOURS,
            max_hours=MAX_STATISTICS_LOOKBACK_HOURS,
            expected_kind="statistics",
            expected_scope_ids=tuple(sorted(statistic_ids)),
        )
        if data.get("cursor") is not None:
            # Cursor carries period and selected fields so pagination stays consistent.
            if cursor.period not in (None, "5minute", "hour", "day"):
                raise RecoverableToolError(INVALID_CURSOR, {})
            period = cast(Literal["5minute", "hour", "day"], cursor.period or "hour")
            requested_types = cast(tuple[StatisticValueType, ...] | None, cursor.statistic_types)
            if requested_types is not None and (
                not requested_types or set(requested_types) - set(STATISTIC_VALUE_TYPES)
            ):
                raise RecoverableToolError(INVALID_CURSOR, {})
        else:
            period = cast(Literal["5minute", "hour", "day"], data["period"])
            requested_types = cast(
                tuple[StatisticValueType, ...] | None,
                tuple(cast(list[str] | None, data.get("types")) or ()) or None,
            )
            cursor = Cursor(
                kind="statistics",
                scope_ids=cursor.scope_ids,
                start=start,
                end=end,
                cutoffs={},
                period=period,
                statistic_types=requested_types,
            )
        query_types: set[StatisticQueryType] = (
            set(cast(tuple[StatisticQueryType, ...], requested_types))
            if requested_types is not None
            else set(_ALL_STAT_QUERY_TYPES)
        )
        result = await source.fetch_statistics(statistic_ids, start, end, period, cast(set[str], query_types))

        budget = max(1, MAX_STATISTICS_ROWS // len(statistic_ids))
        stream_rows = {
            statistic_id: [_statistic_row_to_dict(row, requested_types) for row in rows]
            for statistic_id, rows in result.items()
        }
        pages, next_cutoffs = _paginate_streams(stream_rows, budget=budget, cutoffs=cursor.cutoffs)
        shaped_statistics = {
            statistic_id: {"rows": page, "fields": _statistic_fields(page)} for statistic_id, page in pages.items()
        }

        return _windowed_payload(
            start,
            end,
            {"period": period, "statistics": shaped_statistics},
            Cursor(
                kind="statistics",
                scope_ids=cursor.scope_ids,
                start=start,
                end=end,
                cutoffs=next_cutoffs,
                period=period,
                statistic_types=requested_types,
            )
            if next_cutoffs
            else None,
        )


@final
class GetLogbookTool(_RecorderTool):
    """Return logbook events for visible entities."""

    name = TOOL_GET_LOGBOOK
    description = build_get_logbook_description()
    parameters: vol.Schema = vol.Schema(
        {
            vol.Optional("entity_ids", description="One or up to 20 entity IDs to scope the logbook."): vol.All(
                cv.ensure_list,
                [cv.entity_id],
                vol.Length(min=1, max=MAX_RECORDER_ENTITY_IDS),
            ),
            **_SELECTOR_FIELDS,
            vol.Optional(
                "hours", description="Relative window size in hours; used when start/end are omitted."
            ): _HOURS_ARG,
            vol.Optional("start", description="Window start (ISO-8601). Default now-24h."): _iso_datetime,
            vol.Optional("end", description="Window end (ISO-8601). Default now."): _iso_datetime,
            vol.Optional(
                "cursor",
                description=(
                    "Opaque cursor from this tool's prior next_cursor for the same resolved scope; "
                    "omit start, end, and hours when using it."
                ),
            ): str,
        }
    )

    @override
    async def _query(
        self,
        snapshot: HomeSnapshot,
        data: dict[str, object],
        source: RecorderSource,
    ) -> JsonObjectType:
        entity_ids = resolve_entity_ids(snapshot, data, "entity_ids")
        start, end, cursor = _resolve_window(
            data,
            now=source.now,
            default_hours=DEFAULT_LOGBOOK_WINDOW_HOURS,
            max_hours=MAX_RECORDER_LOOKBACK_HOURS,
            expected_kind="logbook",
            expected_scope_ids=tuple(sorted(entity_ids)),
        )
        if not source.logbook_available:
            raise RecoverableToolError("logbook_unavailable", {})

        entries = await source.fetch_logbook(entity_ids, start, end)
        # A logbook page is one flat stream keyed by the sentinel cutoff.
        page_entries, next_cutoff = paginate_stream(
            entries,
            ts_of=_logbook_when,
            budget=MAX_LOGBOOK_ENTRIES,
            cutoff_iso=cursor.cutoffs.get(_LOGBOOK_CURSOR_KEY),
        )
        return _windowed_payload(
            start,
            end,
            {"entries": page_entries},
            Cursor(
                kind="logbook",
                scope_ids=cursor.scope_ids,
                start=start,
                end=end,
                cutoffs={_LOGBOOK_CURSOR_KEY: next_cutoff},
            )
            if next_cutoff is not None
            else None,
        )
