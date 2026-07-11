from datetime import UTC, datetime
from typing import cast

from custom_components.llm_sandbox.llm_api.tools.automation import (
    AutomationRecord,
    AutomationSource,
    GetAutomationTool,
)
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.runtime import build_fixture_automation_source


async def test_automation_run_query_returns_shared_summary_content_and_runs() -> None:
    fixture = get_home("home_default")
    source = build_fixture_automation_source(fixture.snapshot(), fixture)
    tool = GetAutomationTool("eval")
    target = "automation.living_scene_4f7a"

    summary = await tool.run_query(
        cast(dict[str, object], tool.parameters({"query": "evening living room light"})), source
    )
    record = cast(list[dict[str, object]], summary["automations"])[0]
    assert record["entity_id"] == target
    assert record["state"] == "on"
    assert "content" not in record

    content = await tool.run_query(
        cast(dict[str, object], tool.parameters({"entity_ids": [target], "include": ["content"]})), source
    )
    content_record = cast(list[dict[str, object]], content["automations"])[0]
    assert cast(dict[str, object], content_record["content"])["trigger"] == {
        "platform": "sun",
        "event": "sunset",
    }

    runs = await tool.run_query(
        cast(dict[str, object], tool.parameters({"entity_ids": [target], "include": ["runs"]})), source
    )
    run_rows = cast(list[dict[str, object]], cast(list[dict[str, object]], runs["automations"])[0]["runs"])
    assert run_rows[0]["when"] == "2026-06-28T22:15:00+00:00"
    assert all("trace" not in row for row in run_rows)


async def test_automation_summary_does_not_require_content_projection() -> None:
    tool = GetAutomationTool("eval")

    async def fetch_runs(
        _entity_ids: list[str], _start: datetime, _end: datetime
    ) -> dict[str, list[dict[str, object]]]:
        return {}

    source = AutomationSource(
        now=datetime(2026, 6, 29, 12, tzinfo=UTC),
        available=True,
        content_authorized=True,
        records=(
            AutomationRecord(
                entity_id="automation.opaque_4f7a",
                summary={"entity_id": "automation.opaque_4f7a", "title": "Evening Living Room Lights"},
                search_terms=("Evening Living Room Lights", "Living Room Light"),
                content=None,
            ),
        ),
        fetch_runs=fetch_runs,
    )

    result = await tool.run_query(cast(dict[str, object], tool.parameters({})), source)

    assert result["automations"] == [{"entity_id": "automation.opaque_4f7a", "title": "Evening Living Room Lights"}]


async def test_fixture_without_automation_data_is_unavailable_for_every_projection() -> None:
    fixture = get_home("home_minimal")
    source = build_fixture_automation_source(fixture.snapshot(), fixture)
    tool = GetAutomationTool("eval")

    for args in ({}, {"include": ["content"]}, {"include": ["runs"]}):
        data = cast(dict[str, object], tool.parameters(args))
        result = await tool.run_query(data, source)
        assert result["error"]["key"] == "automation_unavailable"
