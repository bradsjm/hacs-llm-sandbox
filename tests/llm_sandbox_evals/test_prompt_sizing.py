from dataclasses import replace

import pytest
from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from custom_components.llm_sandbox.llm_api.data.history import AGGREGATORS
from custom_components.llm_sandbox.llm_api.prompts import resolve_profile
from custom_components.llm_sandbox.llm_api.tools.recorder import GetHistoryTool
from llm_sandbox_evals.agent_runner import build_agent_tools, render_eval_system_prompt
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.prompts import candidate_prompt_sizes, load_candidates
from llm_sandbox_evals.runtime import build_eval_runtime
from llm_sandbox_evals.schema import CaseContext, EvalCase, Expected
from llm_sandbox_evals.tools import EVAL_SCOPE, apply_scope
from voluptuous_openapi import convert


def test_load_candidates_accepts_profile_candidate() -> None:
    candidates = load_candidates(["profile:standard"], DEFAULT_PROMPT_PROFILE)

    assert len(candidates) == 1
    assert candidates[0].id == "profile:standard"
    assert candidates[0].api_prompt

    candidate = replace(
        candidates[0],
        execute_home_code_description="Candidate code capability. Extra detail.",
        get_history_description="Candidate history capability. Extra detail.",
        get_statistics_description="Candidate statistics capability. Extra detail.",
        get_logbook_description="Candidate logbook capability. Extra detail.",
    )
    case = EvalCase(
        id="candidate-tool-descriptions",
        category="unit",
        home="home_default",
        user_request="Describe the available tools.",
        actions_enabled=False,
        llm_context=CaseContext(),
        expected=Expected(),
    )
    fixture = get_home(case.home)
    runtime = build_eval_runtime(
        case,
        candidate,
        resolve_profile(DEFAULT_PROMPT_PROFILE),
        apply_scope(fixture.snapshot(), EVAL_SCOPE),
        fixture,
    )
    tools = build_agent_tools(runtime)
    prompt = render_eval_system_prompt(runtime, tools)

    assert {tool.name: tool.description for tool in tools} == {
        "execute_home_code": candidate.execute_home_code_description,
        "get_history": candidate.get_history_description,
        "get_statistics": candidate.get_statistics_description,
        "get_logbook": candidate.get_logbook_description,
    }
    for tool in tools:
        assert f"- {tool.name}: {tool.description.split('. ', 1)[0]}." in prompt

    no_logbook_case = replace(case, home="home_minimal")
    no_logbook_fixture = get_home(no_logbook_case.home)
    no_logbook_runtime = build_eval_runtime(
        no_logbook_case,
        candidate,
        resolve_profile(DEFAULT_PROMPT_PROFILE),
        apply_scope(no_logbook_fixture.snapshot(), EVAL_SCOPE),
        no_logbook_fixture,
    )
    no_logbook_tools = build_agent_tools(no_logbook_runtime)
    no_logbook_prompt = render_eval_system_prompt(no_logbook_runtime, no_logbook_tools)

    assert "get_logbook" not in {tool.name for tool in no_logbook_tools}
    assert "Candidate logbook capability." not in no_logbook_prompt


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
