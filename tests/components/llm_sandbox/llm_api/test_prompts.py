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


def test_execute_home_code_description_includes_sql_schema_contract() -> None:
    """The tool description advertises the same query surface as the base prompt."""
    description = build_execute_home_code_description()

    assert "await hass.query(sql, hours=N)" in description
    assert "Tables: states(" in description
    assert "Views: state_history(" in description
    assert "No registry tables" in description
