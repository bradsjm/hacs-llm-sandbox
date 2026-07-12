from datetime import UTC, datetime
from typing import cast

from custom_components.llm_sandbox.llm_api.tools.automation import (
    AutomationRecord,
    AutomationSource,
    GetAutomationTool,
)
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.runtime import build_fixture_automation_source


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
