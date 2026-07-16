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


async def test_full_fixture_automation_source_supports_search_content_and_runs() -> None:
    fixture = get_home("home_full")
    source = build_fixture_automation_source(fixture.snapshot(), fixture)
    tool = GetAutomationTool("eval")
    living_room_id = "automation.living_room_motion_lights"
    server_room_id = "automation.server_room_temperature_protection"
    living_room_content = {
        "id": "living_room_motion_lights",
        "alias": "Living Room Motion Lights",
        "description": "Turn on the Living Room lights when motion is detected.",
        "triggers": [
            {
                "trigger": "state",
                "entity_id": "binary_sensor.living_room_motion",
                "to": "on",
            }
        ],
        "actions": [
            {
                "action": "light.turn_on",
                "target": {
                    "entity_id": [
                        "light.living_room_ceiling",
                        "light.living_room_accent",
                    ]
                },
            }
        ],
        "mode": "single",
    }
    server_room_content = {
        "id": "server_room_temperature_protection",
        "alias": "Server Room Temperature Protection",
        "description": "Turn off the Server Room outlet when its temperature is too high.",
        "triggers": [
            {
                "trigger": "numeric_state",
                "entity_id": "sensor.server_room_temperature",
                "above": 28,
            }
        ],
        "actions": [
            {
                "action": "switch.turn_off",
                "target": {"entity_id": "switch.server_room_outlet"},
            }
        ],
        "mode": "single",
    }

    unfiltered = await tool.run_query(cast(dict[str, object], tool.parameters({})), source)
    assert [record["entity_id"] for record in unfiltered["automations"]] == [
        living_room_id,
        server_room_id,
    ]

    motion_search = await tool.run_query(
        cast(dict[str, object], tool.parameters({"query": "motion lights"})),
        source,
    )
    assert [record["entity_id"] for record in motion_search["automations"]] == [living_room_id]

    server_search = await tool.run_query(
        cast(dict[str, object], tool.parameters({"query": "server room temperature"})),
        source,
    )
    assert server_search["automations"] == [
        {
            "entity_id": server_room_id,
            "title": "Server Room Temperature Protection",
            "state": "on",
            "is_on": True,
            "available": True,
            "description": "Turn off the Server Room outlet when its temperature is too high.",
            "references": {
                "entities": [
                    {"id": "sensor.server_room_temperature", "name": "Server Room Temperature"},
                    {"id": "switch.server_room_outlet", "name": "Server Room Outlet"},
                ]
            },
        }
    ]

    living_content_result = await tool.run_query(
        cast(
            dict[str, object],
            tool.parameters({"entity_ids": [living_room_id], "include": ["content"]}),
        ),
        source,
    )
    assert living_content_result["automations"][0]["content"] == living_room_content

    server_content_result = await tool.run_query(
        cast(
            dict[str, object],
            tool.parameters({"entity_ids": [server_room_id], "include": ["content"]}),
        ),
        source,
    )
    assert server_content_result["automations"][0]["content"] == server_room_content

    living_runs_result = await tool.run_query(
        cast(
            dict[str, object],
            tool.parameters({"entity_ids": [living_room_id], "include": ["runs"], "hours": 1}),
        ),
        source,
    )
    assert living_runs_result["automations"][0]["runs"] == [
        {
            "entity_id": living_room_id,
            "when": "2026-06-29T11:40:00+00:00",
            "name": "Living Room Motion Lights",
            "message": "triggered",
        }
    ]

    server_runs_result = await tool.run_query(
        cast(
            dict[str, object],
            tool.parameters({"entity_ids": [server_room_id], "include": ["runs"], "hours": 4}),
        ),
        source,
    )
    assert server_runs_result["automations"][0]["runs"] == [
        {
            "entity_id": server_room_id,
            "when": "2026-06-29T09:15:00+00:00",
            "name": "Server Room Temperature Protection",
            "message": "triggered",
        }
    ]
