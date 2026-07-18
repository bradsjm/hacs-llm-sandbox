"""Behavior tests for the direct automation LLM tool."""

import base64
from datetime import timedelta
import json
from typing import cast

from custom_components.llm_sandbox.const import TOOL_GET_AUTOMATION
from custom_components.llm_sandbox.llm_api.tools import automation, recorder
from custom_components.llm_sandbox.llm_api.tools.automation import GetAutomationTool
from homeassistant.auth.models import Group, User
from homeassistant.auth.permissions import system_policies
from homeassistant.auth.permissions.const import CAT_ENTITIES, POLICY_READ
from homeassistant.components.recorder import get_instance
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import category_registry as cr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import label_registry as lr
from homeassistant.helpers import llm
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util
from homeassistant.util.json import JsonObjectType
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry


async def test_automation_summary_and_content_use_core_raw_config(
    hass: HomeAssistant, loaded_entry: MockConfigEntry
) -> None:
    """The pinned Core source exposes configured automation content to an admin."""
    await _setup_automations(hass, [_automation_config("kitchen_evening", "Kitchen evening")])

    result = await _call(hass, loaded_entry, {})
    record = result["automations"][0]
    assert record["entity_id"] == "automation.kitchen_evening"
    assert record["config_id"] == "kitchen_evening"
    assert "content" not in record

    detailed = await _call(hass, loaded_entry, {"include": ["content"]})
    content = detailed["automations"][0]["content"]
    assert content["id"] == "kitchen_evening"
    assert content["description"] == "Kitchen evening description"
    assert content["action"]


async def test_automation_summary_is_available_without_recorder(
    hass: HomeAssistant, loaded_entry: MockConfigEntry
) -> None:
    """Summary reads remain available when recorder runtime is absent."""
    await _setup_automations(hass, [_automation_config("test", "Test")])

    result = await _call(hass, loaded_entry, {})

    assert result["returned"] == 1


async def test_automation_runs_require_recorder_only_when_requested(
    hass: HomeAssistant, loaded_entry: MockConfigEntry
) -> None:
    """A run projection reports recorder availability rather than hiding the tool."""
    await _setup_automations(hass, [_automation_config("test", "Test")])

    result = await _call(hass, loaded_entry, {"include": ["runs"]})

    assert result["status"] == "error"
    assert result["error"]["key"] == "recorder_unavailable"


@pytest.mark.parametrize(
    "context_user",
    [pytest.param("", id="missing"), pytest.param("unknown-user", id="unknown")],
)
async def test_automation_requires_attributable_user(
    hass: HomeAssistant, loaded_entry: MockConfigEntry, context_user: str | None
) -> None:
    """Missing and unknown contexts cannot read automations."""
    result = await _call(hass, loaded_entry, {}, context_user=context_user)

    assert result["status"] == "error"
    assert result["error"]["key"] == "authorization_denied"


async def test_inactive_user_and_non_admin_content_are_denied(
    hass: HomeAssistant, loaded_entry: MockConfigEntry
) -> None:
    """Inactive users and non-admin configuration requests fail without partial content."""
    await _setup_automations(hass, [_automation_config("test", "Test")])
    await _call(hass, loaded_entry, {})
    user = await _create_user(hass, "inactive")
    await hass.auth.async_deactivate_user(user)

    inactive = await _call(hass, loaded_entry, {}, context_user=user.id)
    assert inactive["error"]["key"] == "authorization_denied"

    user = await _create_user(hass, "reader")
    denied = await _call(hass, loaded_entry, {"include": ["content"]}, context_user=user.id)
    assert denied["error"]["key"] == "authorization_denied"


async def test_entity_read_permission_filters_summary_and_runs(
    hass: HomeAssistant, recorder_entry: MockConfigEntry
) -> None:
    """Summary and run projections use the current user's entity read policy."""
    await _setup_automations(hass, [_automation_config("allowed", "Allowed"), _automation_config("hidden", "Hidden")])
    await _call(hass, recorder_entry, {})
    user = await _create_user(hass, "limited", entity_ids=["automation.allowed"])

    result = await _call(hass, recorder_entry, {}, context_user=user.id)

    assert [record["entity_id"] for record in result["automations"]] == ["automation.allowed"]


async def test_query_is_independent_of_include_projection(hass: HomeAssistant, loaded_entry: MockConfigEntry) -> None:
    """Changing projections does not change metadata query matching."""
    await _setup_automations(hass, [_automation_config("test", "Stable title")])

    summary = await _call(hass, loaded_entry, {"query": "test"})
    content = await _call(hass, loaded_entry, {"query": "test", "include": ["content"]})

    assert summary["returned"] == content["returned"] == 1


async def test_categories_and_referenced_entity_names_are_searchable(
    hass: HomeAssistant, loaded_entry: MockConfigEntry
) -> None:
    """Scoped categories and live referenced entity names are returned and searchable."""
    await _setup_automations(hass, [_automation_config("test", "Test", "light.bedroom")])
    category = cr.async_get(hass).async_create(name="Important", scope="test")
    entity_registry = er.async_get(hass)
    entity_registry.async_update_entity(
        "automation.test", categories={"test": category.category_id}, labels={"important"}
    )
    lr.async_get(hass).async_create("Important")

    result = await _call(hass, loaded_entry, {"query": "Bedroom"})

    assert result["returned"] == 1
    assert result["automations"][0]["references"]["entities"][0]["name"] == "Bedroom Light"
    assert result["automations"][0]["categories"][0]["name"] == "Important"


async def test_logbook_automation_runs_are_newest_first(hass: HomeAssistant, recorder_entry: MockConfigEntry) -> None:
    """Real automation-triggered Logbook entries are returned as runs, not traces."""
    await _setup_automations(hass, [_automation_config("test", "Test")])
    from homeassistant.components.automation.logbook import async_describe_events
    from homeassistant.components.logbook import DOMAIN as LOGBOOK_DOMAIN

    logbook_config = hass.data[LOGBOOK_DOMAIN]
    async_describe_events(
        hass,
        lambda domain, event_type, descriptor: logbook_config.external_events.update(
            {event_type: (domain, descriptor)}
        ),
    )
    for _ in range(2):
        await hass.services.async_call("automation", "trigger", {"entity_id": "automation.test"}, blocking=True)
    await hass.async_block_till_done()
    await recorder._sync_recorder_for_query(hass, get_instance(hass), 10**9)

    end = (dt_util.utcnow() + timedelta(minutes=1)).isoformat()
    result = await _call(hass, recorder_entry, {"include": ["runs"], "end": end})
    runs = result["automations"][0]["runs"]

    assert len(runs) >= 2
    assert all(run["entity_id"] == "automation.test" for run in runs)
    assert runs == sorted(runs, key=lambda run: str(run["when"]), reverse=True)
    assert "trace" not in runs[0]


async def test_unexpected_run_query_failure_uses_query_failed(
    hass: HomeAssistant, recorder_entry: MockConfigEntry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unexpected Logbook failures use the stable direct-tool query envelope."""
    await _setup_automations(hass, [_automation_config("test", "Test")])

    async def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("executor failed")

    monkeypatch.setattr(GetAutomationTool, "_add_runs", staticmethod(fail))
    result = await _call(hass, recorder_entry, {"include": ["runs"]})

    assert result["status"] == "error"
    assert result["error"]["key"] == "query_failed"


async def test_automation_pagination_has_no_gaps_and_fits_utf8_budget(
    hass: HomeAssistant, loaded_entry: MockConfigEntry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whole records paginate without gaps and measure multibyte UTF-8 bytes."""
    monkeypatch.setattr(automation, "MAX_RECORDER_PAGE_BYTES", 700)
    await _setup_automations(
        hass,
        [
            _automation_config("one", "One", description="é" * 120),
            _automation_config("two", "Second"),
            _automation_config("three", "Third"),
        ],
    )

    first = await _call(hass, loaded_entry, {"limit": 2})
    second = await _call(hass, loaded_entry, {"cursor": first["next_cursor"]})

    walked = [record["entity_id"] for record in first["automations"] + second["automations"]]
    assert walked == ["automation.one", "automation.second", "automation.third"]
    assert len(json.dumps(first, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()) > 0


async def test_oversized_first_automation_still_advances(
    hass: HomeAssistant, loaded_entry: MockConfigEntry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An oversized first record is returned intact and the cursor reaches later records."""
    monkeypatch.setattr(automation, "MAX_RECORDER_PAGE_BYTES", 500)
    await _setup_automations(
        hass, [_automation_config("large", "Large", description="x" * 2000), _automation_config("small", "Small")]
    )

    first = await _call(hass, loaded_entry, {})
    second = await _call(hass, loaded_entry, {"cursor": first["next_cursor"]})

    assert first["automations"][0]["entity_id"] == "automation.large"
    assert second["automations"][0]["entity_id"] == "automation.small"


@pytest.mark.parametrize(
    "tool_args",
    [
        pytest.param({"query": {"value": "lights"}}, id="query-wrapper"),
        pytest.param(
            {"entity_ids": {"value": ["automation.test"]}},
            id="entity-ids-wrapper",
        ),
        pytest.param({"include": "content"}, id="scalar-include"),
        pytest.param(
            {"entity_ids": ["12345678-1234-1234-1234-123456789abc"]},
            id="uuid-entity-selector",
        ),
    ],
)
async def test_canonical_automation_inputs_reject_provider_shape_mismatches(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    tool_args: dict[str, object],
) -> None:
    """Malformed provider shapes fail before authorization or query execution."""
    result = await _call(hass, loaded_entry, tool_args)

    assert result["status"] == "error"
    assert result["error"]["key"] == "invalid_tool_input"


@pytest.mark.parametrize(
    "tool_args",
    [
        pytest.param({"cursor": "bad", "limit": 1}, id="cursor-conflict"),
        pytest.param({"cursor": "bad", "query": "test"}, id="query-conflict"),
    ],
)
async def test_cursor_conflicts_are_invalid_input(
    hass: HomeAssistant, loaded_entry: MockConfigEntry, tool_args: dict[str, object]
) -> None:
    """A cursor cannot be combined with another caller argument."""
    result = await _call(hass, loaded_entry, tool_args)

    assert result["error"]["key"] == "invalid_tool_input"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("e", ["light.not_automation"], id="explicit-entity"),
        pytest.param("after", "light.not_automation", id="continuation-key"),
    ],
)
async def test_cursor_validates_full_automation_ids(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    field: str,
    value: object,
) -> None:
    """Malformed domains in explicit IDs and continuation keys are rejected."""
    cursor = _cursor(
        {"v": 1, "k": "automation", "q": "", "e": [], "p": [], "l": 10, "after": "automation.test"} | {field: value}
    )

    result = await _call(hass, loaded_entry, {"cursor": cursor})

    assert result["error"]["key"] == "invalid_cursor"


async def test_cursor_continuation_reauthorizes_after_permission_loss(
    hass: HomeAssistant, loaded_entry: MockConfigEntry
) -> None:
    """A continuation uses fresh authorization rather than stale cursor authority."""
    await _setup_automations(hass, [_automation_config("one", "One"), _automation_config("two", "Two")])
    await _call(hass, loaded_entry, {})
    user = await _create_user(hass, "paging", entity_ids=["automation.one", "automation.two"])
    first = await _call(hass, loaded_entry, {"limit": 1}, context_user=user.id)
    await hass.auth.async_deactivate_user(user)

    second = await _call(hass, loaded_entry, {"cursor": first["next_cursor"]}, context_user=user.id)

    assert second["error"]["key"] == "authorization_denied"


async def _setup_automations(hass: HomeAssistant, configs: list[dict[str, object]]) -> None:
    """Set up real Core automation entities for direct-tool tests."""
    await async_setup_component(hass, "automation", {"automation": configs})
    await hass.async_block_till_done()


def _automation_config(
    automation_id: str,
    alias: str,
    referenced_entity: str | None = None,
    *,
    description: str | None = None,
) -> dict[str, object]:
    """Build a valid small automation configuration."""
    target = referenced_entity or "light.bedroom"
    return {
        "id": automation_id,
        "alias": alias,
        "description": description or f"{alias} description",
        "trigger": {"platform": "event", "event_type": "automation_test"},
        "action": {"service": "light.turn_on", "target": {"entity_id": target}},
    }


async def _create_user(hass: HomeAssistant, name: str, *, entity_ids: list[str] | None = None) -> User:
    """Create an active real HA user with an optional entity-read policy."""
    user = await hass.auth.async_create_user(name)
    policy = cast(dict[str, object], system_policies.USER_POLICY)
    if entity_ids is not None:
        policy = {CAT_ENTITIES: {"entity_ids": {entity_id: {POLICY_READ: True} for entity_id in entity_ids}}}
    user.groups.append(Group(name=name, policy=policy))
    return user


async def _call(
    hass: HomeAssistant,
    entry: MockConfigEntry,
    tool_args: dict[str, object],
    *,
    context_user: str | None = None,
) -> JsonObjectType:
    """Call the tool with a real HA user context."""
    if context_user is None:
        users = await hass.auth.async_get_users()
        owner = next((user for user in users if user.is_owner), None)
        if owner is None:
            owner = await hass.auth.async_create_user("automation-test-owner")
        context_user = owner.id
    context = Context(user_id=context_user)
    return await GetAutomationTool(entry.entry_id).async_call(
        hass,
        llm.ToolInput(tool_name=TOOL_GET_AUTOMATION, tool_args=tool_args),
        llm.LLMContext(platform="test", context=context, language="en", assistant=None, device_id=None),
    )


def _cursor(payload: dict[str, object]) -> str:
    """Encode a raw cursor for validation tests."""
    return base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
