"""Tests for the prompt profile registry."""

import pytest
from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from custom_components.llm_sandbox.llm_api.prompts import (
    PROFILE_OPTIONS,
    PromptProfile,
    resolve_profile,
)


def test_registry_has_standard_profile_only() -> None:
    assert [p.id for p in PROFILE_OPTIONS] == [DEFAULT_PROMPT_PROFILE]


def test_resolve_standard_profile() -> None:
    profile = resolve_profile(DEFAULT_PROMPT_PROFILE)

    assert isinstance(profile, PromptProfile)
    assert profile.id == DEFAULT_PROMPT_PROFILE
    assert profile.base_prompt


def test_resolve_unknown_profile_raises() -> None:
    with pytest.raises(ValueError, match="unknown prompt profile"):
        resolve_profile("nonexistent")
