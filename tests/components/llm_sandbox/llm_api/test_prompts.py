"""Tests for the prompt profile registry."""

import pytest
from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from custom_components.llm_sandbox.llm_api.prompts import (
    PROFILE_OPTIONS,
    PromptProfile,
    build_execute_home_code_description,
    resolve_profile,
)


def test_registry_has_expected_profiles() -> None:
    assert [p.id for p in PROFILE_OPTIONS] == [DEFAULT_PROMPT_PROFILE, "terse", "minimal"]


def test_resolve_standard_profile() -> None:
    profile = resolve_profile(DEFAULT_PROMPT_PROFILE)

    assert isinstance(profile, PromptProfile)
    assert profile.id == DEFAULT_PROMPT_PROFILE
    assert profile.base_prompt


def test_resolve_unknown_profile_raises() -> None:
    with pytest.raises(ValueError, match="unknown prompt profile"):
        resolve_profile("nonexistent")


@pytest.mark.parametrize("profile", PROFILE_OPTIONS, ids=[profile.id for profile in PROFILE_OPTIONS])
def test_profile_sql_guidance_describes_query_contract(profile: PromptProfile) -> None:
    """Every profile carries accurate compact SQL guidance."""
    assert "await hass.query(sql, hours=N)" in profile.base_prompt
    assert "per-run in-memory database" in profile.base_prompt
    assert "json_extract(attributes" in profile.base_prompt
    assert "registry tables" in profile.base_prompt


@pytest.mark.parametrize("profile", PROFILE_OPTIONS, ids=[profile.id for profile in PROFILE_OPTIONS])
def test_profile_describes_awaitable_facades(profile: PromptProfile) -> None:
    """Profiles distinguish awaitable facades from synchronous reads."""
    assert "Only hass.services.async_call" not in profile.base_prompt
    assert "only awaitable" not in profile.base_prompt.lower()
    assert "hass.history(...)" in profile.base_prompt
    assert "hass.query(...)" in profile.base_prompt
    assert "hass.services.async_call" in profile.base_prompt
    assert "service-catalog reads" in profile.base_prompt
    assert "sync" in profile.base_prompt.lower()


@pytest.mark.parametrize("profile", PROFILE_OPTIONS, ids=[profile.id for profile in PROFILE_OPTIONS])
def test_profile_tool_overview_includes_camera_image(profile: PromptProfile) -> None:
    """Profiles include the full registered tool overview."""
    assert "get_camera_image" in profile.base_prompt


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


def test_execute_home_code_description_includes_sql_schema_contract() -> None:
    """The tool description advertises the same query surface as the base prompt."""
    description = build_execute_home_code_description()

    assert "await hass.query(sql, hours=N)" in description
    assert "Tables: states(" in description
    assert "Views: state_history(" in description
    assert "No registry tables" in description


def test_execute_home_code_description_includes_success_metadata() -> None:
    """The tool description advertises current success envelope metadata."""
    description = build_execute_home_code_description()

    assert "notes" in description
    assert "actions" in description
    assert "resolutions" in description
