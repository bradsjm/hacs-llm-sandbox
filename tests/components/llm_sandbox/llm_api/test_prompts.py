"""Tests for the prompt profile registry."""

from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from custom_components.llm_sandbox.llm_api.prompts import (
    PROFILE_OPTIONS,
    PromptProfile,
    build_execute_home_code_description,
    build_get_history_description,
    build_get_logbook_description,
    build_get_statistics_description,
    compose_system_prompt,
    render_tool_capabilities,
    resolve_profile,
)
from custom_components.llm_sandbox.llm_api.tools.code import ExecuteHomeCodeTool
from custom_components.llm_sandbox.llm_api.tools.recorder import GetHistoryTool, GetLogbookTool, GetStatisticsTool
from custom_components.llm_sandbox.llm_api.tools.vision import GetCameraImageTool
import pytest


def test_registry_has_expected_profiles() -> None:
    assert [p.id for p in PROFILE_OPTIONS] == ["guided", DEFAULT_PROMPT_PROFILE, "frontier"]


def test_resolve_default_profile() -> None:
    profile = resolve_profile(DEFAULT_PROMPT_PROFILE)

    assert isinstance(profile, PromptProfile)
    assert profile.id == DEFAULT_PROMPT_PROFILE
    assert profile.base_prompt


def test_resolve_unknown_profile_raises() -> None:
    with pytest.raises(ValueError, match="unknown prompt profile"):
        resolve_profile("nonexistent")


@pytest.mark.parametrize("profile", PROFILE_OPTIONS, ids=[profile.id for profile in PROFILE_OPTIONS])
def test_profile_sql_guidance_describes_query_contract(profile: PromptProfile) -> None:
    """Every profile carries the full SQL capability contract."""
    assert "await hass.query(sql, hours=N)" in profile.base_prompt
    assert "in-memory database" in profile.base_prompt
    assert "json_extract(attributes" in profile.base_prompt
    assert "registry tables" in profile.base_prompt


@pytest.mark.parametrize("profile", PROFILE_OPTIONS, ids=[profile.id for profile in PROFILE_OPTIONS])
def test_profile_describes_awaitable_facades(profile: PromptProfile) -> None:
    """Profiles distinguish awaitable facades from synchronous reads."""
    assert "Only hass.services.async_call" not in profile.base_prompt
    assert "only awaitable" not in profile.base_prompt.lower()
    assert "hass.history(...)" in profile.base_prompt
    assert "hass.query(...)" in profile.base_prompt
    assert "hass.logbook(...)" in profile.base_prompt
    assert "hass.services.async_call" in profile.base_prompt
    assert "service-catalog reads" in profile.base_prompt
    assert "sync" in profile.base_prompt.lower()


def test_tool_capability_summary_matches_registered_tools() -> None:
    """Generated overview names exactly the registered per-request tools."""
    tools = [
        ExecuteHomeCodeTool("entry-id"),
        GetHistoryTool("entry-id"),
        GetStatisticsTool("entry-id"),
        GetLogbookTool("entry-id"),
        GetCameraImageTool("entry-id"),
    ]

    summary = render_tool_capabilities(tools)

    assert [
        line.split(":", 1)[0].removeprefix("- ")
        for line in summary.splitlines()
        if line.startswith("- ") and ":" in line
    ] == [tool.name for tool in tools]
    assert "get_camera_image" in summary
    assert "direct history, statistics, or logbook retrieval/summarization" in summary
    assert "one execute_home_code call" in summary
    assert "in parallel" in summary
    assert "selectors instead of discovery calls" in summary
    assert "never retrieve the same evidence twice" in summary


@pytest.mark.parametrize("profile", PROFILE_OPTIONS, ids=[profile.id for profile in PROFILE_OPTIONS])
def test_profiles_do_not_hard_code_tool_registry(profile: PromptProfile) -> None:
    """Profiles leave per-request tool naming to the generated registry summary."""
    assert "# LLM Sandbox tools" not in profile.base_prompt


@pytest.mark.parametrize("profile", PROFILE_OPTIONS, ids=[profile.id for profile in PROFILE_OPTIONS])
def test_profile_lists_exposed_snapshot_fields(profile: PromptProfile) -> None:
    """Profiles mention prompt-relevant fields exposed by snapshot records."""
    for field in ("area_id", "floor_id", "device_id", "platform", "unique_id"):
        assert field in profile.base_prompt
    for field in (
        "temperature_unit",
        "length_unit",
        "mass_unit",
        "pressure_unit",
        "volume_unit",
        "area_unit",
        "wind_speed_unit",
        "accumulated_precipitation_unit",
    ):
        assert field in profile.base_prompt
    assert "created_at" in profile.base_prompt
    assert "modified_at" in profile.base_prompt


@pytest.mark.parametrize("profile", PROFILE_OPTIONS, ids=[profile.id for profile in PROFILE_OPTIONS])
def test_profiles_keep_representative_capabilities(profile: PromptProfile) -> None:
    """Profile detail changes coaching, never the supported capability surface."""
    for capability in (
        "frozen visible snapshot",
        "repairs.async_issues",
        "persistent_notifications.async_get_notifications",
        "config_entries.async_entries",
        "async_services_for_target",
        "Builtins:",
        "Imports",
        "No filesystem",
        "event bus",
        "Effective area",
        "HA two-",
    ):
        assert capability in profile.base_prompt


def test_guided_profile_includes_routes_and_compact_examples() -> None:
    prompt = resolve_profile("guided").base_prompt

    for heading in ("### Current state", "### Registry join", "### Composed recorder read", "### Enabled action"):
        assert heading in prompt
    assert "matching standalone tool" in prompt
    assert "Independent direct reads may run in parallel" in prompt
    assert "do not refetch it" in prompt


@pytest.mark.parametrize("profile_id", ["balanced", "frontier"])
def test_balanced_and_frontier_omit_guided_examples(profile_id: str) -> None:
    prompt = resolve_profile(profile_id).base_prompt

    assert "```" not in prompt
    assert "### Current state" not in prompt
    assert "### Registry join" not in prompt


@pytest.mark.parametrize("profile", PROFILE_OPTIONS, ids=[profile.id for profile in PROFILE_OPTIONS])
@pytest.mark.parametrize("actions_enabled", [True, False], ids=["enabled", "disabled"])
def test_profile_composition_retains_action_and_error_contracts(profile: PromptProfile, actions_enabled: bool) -> None:
    prompt = compose_system_prompt(profile, actions_enabled)

    assert prompt.count("## Service calls") == 1
    assert "## Error guidance" in prompt
    assert "guidance" in prompt
    assert "candidates" in prompt
    assert "resolutions" in prompt
    assert "resolved_from" in prompt
    if actions_enabled:
        assert "sequential" in prompt
        assert "no rollback" in prompt
        assert "target" in prompt
        assert "response capture" in prompt
        assert "return_response=True requires" not in prompt
    else:
        assert "async_call is rejected" in prompt
        assert "service-catalog reads" in prompt


def test_empty_candidate_base_prompt_override_is_preserved() -> None:
    """An optimized eval candidate may intentionally replace the base prompt with empty text."""
    profile = resolve_profile("balanced")

    prompt = compose_system_prompt(profile, actions_enabled=False, base_prompt="", tool_section="# Candidate tools")

    assert prompt.startswith("# Candidate tools\n\n## Service calls (disabled)")
    assert profile.base_prompt not in prompt


def test_execute_home_code_description_includes_sql_schema_contract() -> None:
    """The tool description advertises the same query surface as the base prompt."""
    description = build_execute_home_code_description()

    assert "await hass.query(sql, hours=N)" in description
    assert "Tables: states(" in description
    assert "Views: state_history(" in description
    assert "No registry tables" in description
    assert "hass.logbook(...)" in description


def test_execute_home_code_description_includes_success_metadata() -> None:
    """The tool description advertises current success envelope metadata."""
    description = build_execute_home_code_description()

    assert "notes" in description
    assert "actions" in description
    assert "resolutions" in description
    assert "overflow" in description


@pytest.mark.parametrize(
    "description",
    [
        pytest.param(build_get_history_description(), id="history"),
        pytest.param(build_get_statistics_description(), id="statistics"),
        pytest.param(build_get_logbook_description(), id="logbook"),
    ],
)
def test_standalone_recorder_descriptions_route_dependent_work_to_code(description: str) -> None:
    """Standalone recorder descriptions direct dependent work to one code call."""
    assert "Prefer this standalone tool for direct retrieval" in description
    assert "one execute_home_code call for dependent composition or actions" in description
