import pytest
from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from custom_components.llm_sandbox.llm_api.data.history import AGGREGATORS
from custom_components.llm_sandbox.llm_api.tools.recorder import GetHistoryTool
from llm_sandbox_evals.prompts import candidate_prompt_sizes, load_candidates
from voluptuous_openapi import convert


def test_load_candidates_accepts_profile_candidate() -> None:
    candidates = load_candidates(["profile:standard"], DEFAULT_PROMPT_PROFILE)

    assert len(candidates) == 1
    assert candidates[0].id == "profile:standard"
    assert candidates[0].api_prompt


@pytest.mark.parametrize("profile_id", ["terse", "minimal"])
def test_condensed_profiles_load_and_are_smaller_than_standard(profile_id: str) -> None:
    standard = load_candidates(["profile:standard"], DEFAULT_PROMPT_PROFILE)[0]
    terse = load_candidates(["profile:terse"], DEFAULT_PROMPT_PROFILE)[0]
    minimal = load_candidates(["profile:minimal"], DEFAULT_PROMPT_PROFILE)[0]
    candidate = load_candidates([f"profile:{profile_id}"], DEFAULT_PROMPT_PROFILE)[0]

    standard_api_chars, _standard_authored_chars = candidate_prompt_sizes(standard)
    terse_api_chars, _terse_authored_chars = candidate_prompt_sizes(terse)
    minimal_api_chars, _minimal_authored_chars = candidate_prompt_sizes(minimal)
    candidate_api_chars, _candidate_authored_chars = candidate_prompt_sizes(candidate)

    assert candidate.id == f"profile:{profile_id}"
    assert candidate.api_prompt
    assert candidate_api_chars < standard_api_chars
    assert minimal_api_chars < terse_api_chars < standard_api_chars


def test_load_candidates_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="unknown prompt profile"):
        load_candidates(["profile:bogus"], DEFAULT_PROMPT_PROFILE)


def test_get_history_function_schema_exposes_aggregate_filters() -> None:
    parameters = convert(GetHistoryTool("eval").parameters)

    properties = parameters["properties"]
    assert set(AGGREGATORS)
    assert properties["aggregate"]["type"] == "object"
    assert properties["from_state"]["type"] == "string"
    assert properties["to_state"]["type"] == "string"
    assert properties["group_by"]["items"] == {"type": "string"}
    assert properties["bucket"]["type"] == "string"
    assert properties["where"]["items"]["type"] == "object"
    assert properties["order_by"]["type"] == "string"
    assert properties["limit"]["minimum"] == 1
