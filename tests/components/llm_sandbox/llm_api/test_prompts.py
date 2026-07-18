"""Tests for the prompt profile registry."""

from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from custom_components.llm_sandbox.llm_api.prompts import PROFILE_OPTIONS, PromptProfile, resolve_profile
import pytest


def test_registry_has_expected_profiles() -> None:
    assert [p.id for p in PROFILE_OPTIONS] == ["guided", DEFAULT_PROMPT_PROFILE, "frontier"]


def test_resolve_default_profile() -> None:
    profile = resolve_profile(DEFAULT_PROMPT_PROFILE)

    assert isinstance(profile, PromptProfile)
    assert profile.id == DEFAULT_PROMPT_PROFILE
    assert profile.base_prompt


def test_resolve_unknown_profile_raises() -> None:
    with pytest.raises(ValueError, match=r".+"):
        resolve_profile("nonexistent")


@pytest.mark.parametrize("profile", PROFILE_OPTIONS, ids=[profile.id for profile in PROFILE_OPTIONS])
def test_profiles_describe_callable_sandbox_surface(profile: PromptProfile) -> None:
    """Every profile exposes the callable facade surface an LLM must use."""
    for callable_surface in (
        "hass.history(...)",
        "hass.query(...)",
        "hass.logbook(...)",
        "hass.energy(...)",
        "hass.services.async_call",
    ):
        assert callable_surface in profile.base_prompt
