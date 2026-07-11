import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import ModuleType
from typing import cast

from custom_components.llm_sandbox.const import (
    DEFAULT_PROMPT_PROFILE,
    MAX_HISTORY_STATES,
    MAX_LOGBOOK_ENTRIES,
    MAX_RECORDER_PAGE_BYTES,
)
from custom_components.llm_sandbox.llm_api.prompts.profiles import resolve_profile
from custom_components.llm_sandbox.llm_api.tools import recorder
from custom_components.llm_sandbox.llm_api.tools._cursor import Cursor, decode_cursor, encode_cursor
from custom_components.llm_sandbox.llm_api.tools._recorder_runtime import (
    HistoryRowStream,
    LogbookEntryStream,
    StatisticsRowStream,
)
from custom_components.llm_sandbox.llm_api.tools.recorder import GetHistoryTool, GetLogbookTool, GetStatisticsTool
from custom_components.llm_sandbox.snapshot.models import HomeSnapshot
from freezegun.api import FrozenDateTimeFactory
from homeassistant.helpers import llm
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.prompts import baseline_candidate
from llm_sandbox_evals.runtime import build_eval_runtime, build_fixture_recorder_source
from llm_sandbox_evals.schema import CaseContext, EvalCase, Expected
from llm_sandbox_evals.tools import EVAL_SCOPE, apply_scope

_CREATED_AT = datetime(2026, 6, 29, 12, tzinfo=UTC)
_LIVING_TEMP = "sensor.living_temp"
_LIVING_LIGHT = "light.living"


async def test_history_run_query_uses_production_aggregate_shape() -> None:
    snapshot = _snapshot()
    source = build_fixture_recorder_source(snapshot, get_home("home_default"))
    tool = GetHistoryTool("eval")
    data = cast(
        dict[str, object],
        tool.parameters({"entity_ids": [_LIVING_TEMP], "hours": 24, "aggregate": "state_counts"}),
    )

    result = await tool.run_query(snapshot, data, source)

    assert result == {
        "window": {"start": "2026-06-28T12:00:00+00:00", "end": "2026-06-29T12:00:00+00:00"},
        "mode": "state_counts",
        "summary": {_LIVING_TEMP: {"state_counts": {"24.4": 1, "24.9": 1, "25.2": 1}}},
    }


async def test_history_run_query_cursor_round_trip_uses_production_pagination() -> None:
    timestamps = _ascending_timestamps(MAX_HISTORY_STATES + 5)
    fixture = _fixture({"history": {_LIVING_TEMP: _history_rows(timestamps)}, "statistics": {}, "logbook": {}})
    snapshot = _snapshot()
    source = build_fixture_recorder_source(snapshot, fixture)
    tool = GetHistoryTool("eval")
    first_data = cast(
        dict[str, object],
        tool.parameters({"entity_ids": [_LIVING_TEMP], "start": timestamps[0], "end": timestamps[-1]}),
    )

    first_result = await tool.run_query(snapshot, first_data, source)

    first_rows = _history_result_rows(first_result, _LIVING_TEMP)
    assert first_rows
    assert len(first_rows) <= MAX_HISTORY_STATES
    assert _compact_json_bytes(first_result) <= MAX_RECORDER_PAGE_BYTES
    assert first_rows[-1] == [timestamps[-1], str(len(timestamps) - 1)]
    cursor = cast(str, first_result["next_cursor"])
    second_data = cast(dict[str, object], tool.parameters({"entity_ids": [_LIVING_TEMP], "cursor": cursor}))
    second_result = await tool.run_query(snapshot, second_data, source)
    second_rows = _history_result_rows(second_result, _LIVING_TEMP)
    assert second_rows
    assert _compact_json_bytes(second_result) <= MAX_RECORDER_PAGE_BYTES
    assert str(second_rows[-1][0]) < str(first_rows[0][0])
    assert {row[0] for row in first_rows}.isdisjoint(row[0] for row in second_rows)


async def test_logbook_source_injects_entity_id_and_production_paginates() -> None:
    timestamps = _ascending_timestamps(MAX_LOGBOOK_ENTRIES + 3)
    fixture = _fixture(
        {
            "history": {},
            "statistics": {},
            "logbook": {_LIVING_LIGHT: _logbook_rows(timestamps, message_prefix="é" * 500)},
        }
    )
    snapshot = _snapshot()
    source = build_fixture_recorder_source(snapshot, fixture)
    tool = GetLogbookTool("eval")
    data = cast(
        dict[str, object],
        tool.parameters({"entity_ids": [_LIVING_LIGHT], "start": timestamps[0], "end": timestamps[-1]}),
    )

    result = await tool.run_query(snapshot, data, source)

    entries = cast(list[dict[str, object]], result["entries"])
    assert entries
    assert len(entries) <= MAX_LOGBOOK_ENTRIES
    assert _compact_json_bytes(result) <= MAX_RECORDER_PAGE_BYTES
    assert {entry["entity_id"] for entry in entries} == {_LIVING_LIGHT}
    assert entries[-1]["when"] == timestamps[-1]
    second_data = cast(
        dict[str, object],
        tool.parameters({"entity_ids": [_LIVING_LIGHT], "cursor": result["next_cursor"]}),
    )
    second = await tool.run_query(snapshot, second_data, source)
    second_entries = cast(list[dict[str, object]], second["entries"])
    assert second_entries
    assert _compact_json_bytes(second) <= MAX_RECORDER_PAGE_BYTES
    assert str(second_entries[-1]["when"]) < str(entries[0]["when"])
    assert {entry["when"] for entry in entries}.isdisjoint(entry["when"] for entry in second_entries)


async def test_statistics_pagination_fits_utf8_rows_and_advances() -> None:
    """Statistics pages fit the full UTF-8 envelope before the row ceiling."""
    timestamps = _ascending_timestamps(20)
    fixture = _fixture(
        {
            "history": {},
            "statistics": {_LIVING_TEMP: [{"start": timestamp, "state": "é" * 1_500} for timestamp in timestamps]},
            "logbook": {},
        }
    )
    snapshot = _snapshot()
    base_source = build_fixture_recorder_source(snapshot, fixture)
    fetches: list[tuple[list[str], datetime]] = []

    async def fetch_statistics(
        statistic_ids: list[str],
        start: datetime,
        end: datetime,
        period: str,
        types: set[str],
    ) -> StatisticsRowStream:
        fetches.append((statistic_ids, end))
        return await base_source.fetch_statistics(statistic_ids, start, end, period, types)

    source = replace(base_source, fetch_statistics=fetch_statistics)
    tool = GetStatisticsTool("eval")
    first_data = cast(
        dict[str, object],
        tool.parameters(
            {
                "statistic_ids": [_LIVING_TEMP],
                "start": timestamps[0],
                "end": timestamps[-1],
                "types": ["state"],
            }
        ),
    )

    first = await tool.run_query(snapshot, first_data, source)

    first_rows = _statistics_result_rows(first, _LIVING_TEMP)
    assert first_rows
    assert _compact_json_bytes(first) <= MAX_RECORDER_PAGE_BYTES
    assert first_rows[-1][0] == timestamps[-1]
    second_data = cast(
        dict[str, object],
        tool.parameters({"statistic_ids": [_LIVING_TEMP], "cursor": first["next_cursor"]}),
    )
    second = await tool.run_query(snapshot, second_data, source)
    second_rows = _statistics_result_rows(second, _LIVING_TEMP)
    assert second_rows
    assert _compact_json_bytes(second) <= MAX_RECORDER_PAGE_BYTES
    assert str(second_rows[-1][0]) < str(first_rows[0][0])
    assert {row[0] for row in first_rows}.isdisjoint(row[0] for row in second_rows)
    assert fetches[1] == ([_LIVING_TEMP], datetime.fromisoformat(cast(str, first_rows[0][0])))


async def test_statistics_oversized_first_row_is_intact_and_cursor_advances() -> None:
    """An indivisible oversized statistics row is returned alone before older rows."""
    timestamps = _ascending_timestamps(2)
    fixture = _fixture(
        {
            "history": {},
            "statistics": {
                _LIVING_TEMP: [
                    {"start": timestamps[0], "state": "older"},
                    {"start": timestamps[1], "state": "é" * (MAX_RECORDER_PAGE_BYTES + 1)},
                ]
            },
            "logbook": {},
        }
    )
    snapshot = _snapshot()
    source = build_fixture_recorder_source(snapshot, fixture)
    tool = GetStatisticsTool("eval")
    first_data = cast(
        dict[str, object],
        tool.parameters(
            {
                "statistic_ids": [_LIVING_TEMP],
                "start": timestamps[0],
                "end": timestamps[-1],
                "types": ["state"],
            }
        ),
    )

    first = await tool.run_query(snapshot, first_data, source)
    second_data = cast(
        dict[str, object],
        tool.parameters({"statistic_ids": [_LIVING_TEMP], "cursor": first["next_cursor"]}),
    )
    second = await tool.run_query(snapshot, second_data, source)

    assert _statistics_result_rows(first, _LIVING_TEMP) == [
        [timestamps[1], {"state": "é" * (MAX_RECORDER_PAGE_BYTES + 1)}]
    ]
    assert _compact_json_bytes(first) > MAX_RECORDER_PAGE_BYTES
    assert _statistics_result_rows(second, _LIVING_TEMP) == [[timestamps[0], {"state": "older"}]]


async def test_history_byte_cursor_preserves_exhausted_stream_and_narrows_fetch() -> None:
    """A byte-limited stream does not restart a sibling that already exhausted."""
    timestamps = _ascending_timestamps(3)
    fixture = _fixture(
        {
            "history": {
                _LIVING_TEMP: _history_rows([timestamps[2]]),
                _LIVING_LIGHT: [
                    {
                        "state": "é" * 5_000,
                        "attributes": {},
                        "last_changed": timestamp,
                        "last_updated": timestamp,
                    }
                    for timestamp in timestamps[:2]
                ],
            },
            "statistics": {},
            "logbook": {},
        }
    )
    snapshot = _snapshot()
    base_source = build_fixture_recorder_source(snapshot, fixture)
    fetches: list[tuple[list[str], datetime]] = []

    async def fetch_history(entity_ids: list[str], start: datetime, end: datetime) -> HistoryRowStream:
        fetches.append((entity_ids, end))
        return await base_source.fetch_history(entity_ids, start, end)

    source = replace(base_source, fetch_history=fetch_history)
    tool = GetHistoryTool("eval")
    entity_ids = [_LIVING_TEMP, _LIVING_LIGHT]
    first_data = cast(
        dict[str, object],
        tool.parameters({"entity_ids": entity_ids, "start": timestamps[0], "end": timestamps[-1]}),
    )

    first = await tool.run_query(snapshot, first_data, source)
    cursor = decode_cursor(first["next_cursor"], expected_kind="history", expected_scope_ids=tuple(sorted(entity_ids)))
    second_data = cast(dict[str, object], tool.parameters({"entity_ids": entity_ids, "cursor": first["next_cursor"]}))
    second = await tool.run_query(snapshot, second_data, source)

    assert cursor.cutoffs == {_LIVING_TEMP: "", _LIVING_LIGHT: timestamps[1]}
    assert _history_result_rows(first, _LIVING_TEMP) == [[timestamps[2], "0"]]
    assert _history_result_rows(second, _LIVING_TEMP) == []
    assert _history_result_rows(second, _LIVING_LIGHT) == [[timestamps[0], "é" * 5_000]]
    assert fetches[1] == ([_LIVING_LIGHT], datetime.fromisoformat(timestamps[1]))


def test_continuation_query_groups_partition_streams_by_effective_end() -> None:
    """Continuation groups coalesce equal cutoffs and isolate every other safe end."""
    original_end = _CREATED_AT
    newer_cutoff = (_CREATED_AT - timedelta(minutes=1)).isoformat()
    older_cutoff = (_CREATED_AT - timedelta(minutes=2)).isoformat()

    assert recorder._continuation_query_groups(
        ["one", "two"], {"one": newer_cutoff, "two": newer_cutoff}, original_end
    ) == [(["one", "two"], datetime.fromisoformat(newer_cutoff))]
    assert recorder._continuation_query_groups(
        ["one", "two"], {"one": newer_cutoff, "two": older_cutoff}, original_end
    ) == [
        (["one"], datetime.fromisoformat(newer_cutoff)),
        (["two"], datetime.fromisoformat(older_cutoff)),
    ]
    assert recorder._continuation_query_groups(["one", "two"], {"one": older_cutoff}, original_end) == [
        (["one"], datetime.fromisoformat(older_cutoff)),
        (["two"], original_end),
    ]
    assert recorder._continuation_query_groups(["one", "two"], {"one": "", "two": newer_cutoff}, original_end) == [
        (["two"], datetime.fromisoformat(newer_cutoff))
    ]
    assert recorder._continuation_query_groups(["one"], {"one": "not-a-timestamp"}, original_end) == [
        (["one"], original_end)
    ]


async def test_history_continuation_groups_fetches_and_merges_rows() -> None:
    """Distinct stream cutoffs issue sequential grouped reads that merge one response."""
    timestamps = _ascending_timestamps(4)
    entity_ids = [_LIVING_TEMP, _LIVING_LIGHT]
    fixture = _fixture(
        {
            "history": {
                _LIVING_TEMP: _history_rows(timestamps),
                _LIVING_LIGHT: _history_rows(timestamps),
            },
            "statistics": {},
            "logbook": {},
        }
    )
    snapshot = _snapshot()
    base_source = build_fixture_recorder_source(snapshot, fixture)
    fetches: list[tuple[list[str], datetime]] = []

    async def fetch_history(entity_ids: list[str], start: datetime, end: datetime) -> HistoryRowStream:
        fetches.append((entity_ids, end))
        return await base_source.fetch_history(entity_ids, start, end)

    source = replace(base_source, fetch_history=fetch_history)
    cursor = encode_cursor(
        Cursor(
            kind="history",
            scope_ids=tuple(sorted(entity_ids)),
            start=datetime.fromisoformat(timestamps[0]),
            end=datetime.fromisoformat(timestamps[-1]),
            cutoffs={_LIVING_TEMP: timestamps[-1], _LIVING_LIGHT: timestamps[-2]},
        )
    )
    tool = GetHistoryTool("eval")
    data = cast(dict[str, object], tool.parameters({"entity_ids": entity_ids, "cursor": cursor}))

    result = await tool.run_query(snapshot, data, source)

    assert fetches == [
        ([_LIVING_TEMP], datetime.fromisoformat(timestamps[-1])),
        ([_LIVING_LIGHT], datetime.fromisoformat(timestamps[-2])),
    ]
    assert _history_result_rows(result, _LIVING_TEMP) == [
        [timestamp, str(index)] for index, timestamp in enumerate(timestamps[:-1])
    ]
    assert _history_result_rows(result, _LIVING_LIGHT) == [
        [timestamp, str(index)] for index, timestamp in enumerate(timestamps[:-2])
    ]


async def test_history_byte_cursor_walk_returns_each_unique_timestamp_once() -> None:
    """Successive byte-limited cursor pages cover the complete unique-timestamp stream."""
    timestamps = _ascending_timestamps(4)
    fixture = _fixture(
        {
            "history": {
                _LIVING_TEMP: [
                    {
                        "state": "é" * 5_000,
                        "attributes": {},
                        "last_changed": timestamp,
                        "last_updated": timestamp,
                    }
                    for timestamp in timestamps
                ]
            },
            "statistics": {},
            "logbook": {},
        }
    )
    snapshot = _snapshot()
    source = build_fixture_recorder_source(snapshot, fixture)
    tool = GetHistoryTool("eval")
    data = cast(
        dict[str, object],
        tool.parameters({"entity_ids": [_LIVING_TEMP], "start": timestamps[0], "end": timestamps[-1]}),
    )
    seen: list[str] = []

    while True:
        result = await tool.run_query(snapshot, data, source)
        seen.extend(str(row[0]) for row in _history_result_rows(result, _LIVING_TEMP))
        if "next_cursor" not in result:
            break
        data = cast(
            dict[str, object], tool.parameters({"entity_ids": [_LIVING_TEMP], "cursor": result["next_cursor"]})
        )

    assert len(seen) == len(set(seen)) == len(timestamps)
    assert set(seen) == set(timestamps)


async def test_logbook_cursor_narrows_continuation_fetch() -> None:
    """Logbook continuation queries retain the original window but fetch only older rows."""
    timestamps = _ascending_timestamps(2)
    fixture = _fixture(
        {
            "history": {},
            "statistics": {},
            "logbook": {_LIVING_LIGHT: _logbook_rows(timestamps, message_prefix="é" * 5_000)},
        }
    )
    snapshot = _snapshot()
    base_source = build_fixture_recorder_source(snapshot, fixture)
    fetches: list[datetime] = []

    async def fetch_logbook(entity_ids: list[str], start: datetime, end: datetime) -> LogbookEntryStream:
        fetches.append(end)
        return await base_source.fetch_logbook(entity_ids, start, end)

    source = replace(base_source, fetch_logbook=fetch_logbook)
    tool = GetLogbookTool("eval")
    first_data = cast(
        dict[str, object],
        tool.parameters({"entity_ids": [_LIVING_LIGHT], "start": timestamps[0], "end": timestamps[-1]}),
    )

    first = await tool.run_query(snapshot, first_data, source)
    second_data = cast(
        dict[str, object], tool.parameters({"entity_ids": [_LIVING_LIGHT], "cursor": first["next_cursor"]})
    )
    second = await tool.run_query(snapshot, second_data, source)

    assert cast(list[dict[str, object]], second["entries"])[0]["when"] == timestamps[0]
    assert fetches[1] == datetime.fromisoformat(timestamps[1])


async def test_statistics_duplicate_types_are_stably_deduplicated() -> None:
    """Repeated requested statistic fields do not inflate rows or the continuation cursor."""
    timestamps = _ascending_timestamps(2)
    fixture = _fixture(
        {
            "history": {},
            "statistics": {
                _LIVING_TEMP: [
                    {"start": timestamps[0], "sum": 1.0, "state": "é" * 5_000, "mean": 2.0},
                    {"start": timestamps[1], "sum": 3.0, "state": "é" * 5_000, "mean": 4.0},
                ]
            },
            "logbook": {},
        }
    )
    snapshot = _snapshot()
    source = build_fixture_recorder_source(snapshot, fixture)
    tool = GetStatisticsTool("eval")
    requested_types = ["sum", "state", "sum", "mean", "state"]
    data = cast(
        dict[str, object],
        tool.parameters(
            {
                "statistic_ids": [_LIVING_TEMP],
                "start": timestamps[0],
                "end": timestamps[-1],
                "types": requested_types,
            }
        ),
    )

    result = await tool.run_query(snapshot, data, source)
    cursor = decode_cursor(result["next_cursor"], expected_kind="statistics", expected_scope_ids=(_LIVING_TEMP,))
    second_data = cast(
        dict[str, object], tool.parameters({"statistic_ids": [_LIVING_TEMP], "cursor": result["next_cursor"]})
    )
    await tool.run_query(snapshot, second_data, source)
    row = _statistics_result_rows(result, _LIVING_TEMP)[0]

    assert cursor.statistic_types == ("sum", "state", "mean")
    assert list(cast(dict[str, object], row[1])) == ["sum", "state", "mean"]


async def test_execute_home_code_runs_with_eval_runtime_context() -> None:
    case = _case()
    fixture = get_home("home_default")
    snapshot = apply_scope(_snapshot(), EVAL_SCOPE, anchor_device_id=case.llm_context.device_id)
    runtime = build_eval_runtime(
        case, baseline_candidate(), resolve_profile(DEFAULT_PROMPT_PROFILE), snapshot, fixture
    )
    data = cast(
        dict[str, object],
        runtime.code_tool.parameters(
            {
                "code": "result = await hass.query(\"select entity_id, state from states where entity_id = 'light.living'\")"
            }
        ),
    )

    result = await runtime.code_tool.run_execute(
        snapshot,
        data,
        llm.LLMContext("test", None, "en", None, None),
        runtime.runtime_context_factory(),
    )

    assert result["execution"] == {"status": "ok"}
    assert result["output"] == [{"entity_id": "light.living", "state": "on"}]


async def test_execute_home_code_logbook_uses_fresh_fixture_runtime_seam(freezer: FrozenDateTimeFactory) -> None:
    freezer.move_to(_CREATED_AT)
    case = _case()
    fixture = get_home("home_default")
    snapshot = apply_scope(_snapshot(), EVAL_SCOPE, anchor_device_id=case.llm_context.device_id)
    runtime = build_eval_runtime(
        case, baseline_candidate(), resolve_profile(DEFAULT_PROMPT_PROFILE), snapshot, fixture
    )
    data = cast(
        dict[str, object],
        runtime.code_tool.parameters({"code": 'result = await hass.logbook("light.living", hours=24)'}),
    )

    result = await runtime.code_tool.run_execute(
        snapshot,
        data,
        llm.LLMContext("test", None, "en", None, None),
        runtime.runtime_context_factory(),
    )

    assert result["execution"] == {"status": "ok"}
    assert result["output"] == [
        {
            "entity_id": "light.living",
            "when": "2026-06-29T08:00:00+00:00",
            "name": "Living Room Light",
            "message": "turned off",
        },
        {
            "entity_id": "light.living",
            "when": "2026-06-29T11:30:00+00:00",
            "name": "Living Room Light",
            "message": "turned on",
        },
    ]


def _case() -> EvalCase:
    return EvalCase(
        id="production-core-unit",
        category="unit",
        home="home_default",
        user_request="exercise production core",
        actions_enabled=False,
        llm_context=CaseContext(),
        expected=Expected(),
    )


def _snapshot() -> HomeSnapshot:
    return cast(HomeSnapshot, get_home("home_default").snapshot())


def _fixture(recorder_data: dict[str, object]) -> ModuleType:
    module = ModuleType("fixture_home")

    def recorder() -> dict[str, object]:
        return recorder_data

    module.__dict__["recorder"] = recorder
    return module


def _ascending_timestamps(count: int) -> list[str]:
    start = _CREATED_AT - timedelta(minutes=count - 1)
    return [(start + timedelta(minutes=index)).isoformat() for index in range(count)]


def _history_rows(timestamps: list[str]) -> list[dict[str, object]]:
    return [
        {
            "state": str(index),
            "attributes": {"unit_of_measurement": "°C"},
            "last_changed": timestamp,
            "last_updated": timestamp,
        }
        for index, timestamp in enumerate(timestamps)
    ]


def _logbook_rows(timestamps: list[str], *, message_prefix: str = "changed state") -> list[dict[str, object]]:
    return [
        {"when": timestamp, "name": "Living Room Light", "message": f"{message_prefix} {index}"}
        for index, timestamp in enumerate(timestamps)
    ]


def _history_result_rows(result: dict[str, object], entity_id: str) -> list[list[object]]:
    entities = cast(dict[str, dict[str, object]], result["entities"])
    return cast(list[list[object]], entities[entity_id]["rows"])


def _statistics_result_rows(result: dict[str, object], statistic_id: str) -> list[list[object]]:
    statistics = cast(dict[str, dict[str, object]], result["statistics"])
    return cast(list[list[object]], statistics[statistic_id]["rows"])


def _compact_json_bytes(result: dict[str, object]) -> int:
    return len(json.dumps(result, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8"))
