"""Prompt and function-schema assembly for the dev-only eval harness."""

import json
from dataclasses import replace
from pathlib import Path

from custom_components.llm_sandbox.const import (
    DEFAULT_PROMPT_PROFILE,
)
from custom_components.llm_sandbox.llm_api.prompts import (
    build_execute_home_code_description,
    build_get_history_description,
    build_get_logbook_description,
    build_get_statistics_description,
    resolve_profile,
)

from llm_sandbox_evals.schema import PromptCandidate

_BASELINE_ID = "baseline"


def candidate_prompt_sizes(candidate: PromptCandidate) -> tuple[int, int]:
    """Return ``(api_prompt_chars, authored_prompt_chars)`` for a candidate.

    ``api_prompt_chars`` is the length of the COPRO-mutable instruction.
    ``authored_prompt_chars`` adds the four tool-description fields. Action
    sections and JSON-schema scaffolding are excluded because they are
    candidate-invariant (identical across candidates) and only add a constant
    offset, so they would dilute size-comparison signal.
    """
    api_prompt_chars = len(candidate.api_prompt)
    authored_prompt_chars = api_prompt_chars + (
        len(candidate.execute_home_code_description)
        + len(candidate.get_history_description)
        + len(candidate.get_statistics_description)
        + len(candidate.get_logbook_description)
    )
    return api_prompt_chars, authored_prompt_chars


def baseline_candidate(prompt_profile_id: str = DEFAULT_PROMPT_PROFILE) -> PromptCandidate:
    """Return the production-baseline prompt candidate."""
    profile = resolve_profile(prompt_profile_id)
    return PromptCandidate(
        id=_BASELINE_ID,
        api_prompt=profile.base_prompt,
        execute_home_code_description=build_execute_home_code_description(),
        get_history_description=build_get_history_description(),
        get_statistics_description=build_get_statistics_description(),
        get_logbook_description=build_get_logbook_description(),
    )


def load_candidates(candidate_ids: list[str], prompt_profile_id: str) -> list[PromptCandidate]:
    """Load supported prompt candidates, rejecting unknown candidate ids."""
    candidates: list[PromptCandidate] = []
    for candidate_id in candidate_ids:
        # Branch boundary: the production baseline is the built-in candidate.
        if candidate_id == _BASELINE_ID:
            candidates.append(baseline_candidate(prompt_profile_id))
            continue
        # Branch boundary: optimizer artifacts are explicitly loaded from JSON.
        if candidate_id.startswith("optimized:"):
            candidates.append(_load_optimized(candidate_id.removeprefix("optimized:")))
            continue
        # Branch boundary: any registered production profile is loadable as a candidate.
        if candidate_id.startswith("profile:"):
            profile_id = candidate_id.removeprefix("profile:")
            # resolve_profile (called inside baseline_candidate) raises ValueError for unknown ids.
            candidates.append(replace(baseline_candidate(profile_id), id=candidate_id))
            continue
        # Candidate ids are configuration errors when unknown; do not silently fall back.
        raise ValueError(f"unknown prompt candidate id(s): {candidate_id}")
    return candidates


def _load_optimized(path: str) -> PromptCandidate:
    """Load an optimized prompt candidate from a JSON artifact."""
    decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("optimized candidate JSON must contain an object")
    return PromptCandidate(
        id=_string_field(decoded, "id"),
        api_prompt=_string_field(decoded, "api_prompt"),
        execute_home_code_description=_string_field(decoded, "execute_home_code_description"),
        get_history_description=_string_field(decoded, "get_history_description"),
        get_statistics_description=_string_field(decoded, "get_statistics_description"),
        get_logbook_description=_string_field(decoded, "get_logbook_description"),
    )


def _string_field(data: dict[object, object], key: str) -> str:
    """Return a required string field from an optimizer candidate artifact."""
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"optimized candidate field {key!r} must be a string")
    return value
