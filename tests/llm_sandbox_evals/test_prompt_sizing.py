from dataclasses import replace

from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from custom_components.llm_sandbox.llm_api.data.history import AGGREGATORS
from custom_components.llm_sandbox.llm_api.prompts import resolve_profile
from custom_components.llm_sandbox.llm_api.tools.recorder import GetHistoryTool
from llm_sandbox_evals.agent_runner import build_agent_tools, render_eval_system_prompt
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.prompts import candidate_prompt_sizes, load_candidates
from llm_sandbox_evals.runtime import build_eval_runtime
from llm_sandbox_evals.schema import EvalCase, RequiredAction
from llm_sandbox_evals.tools import EVAL_SCOPE, apply_scope
import pytest
from voluptuous_openapi import convert


def test_load_candidates_accepts_profile_candidate() -> None:
    candidates = load_candidates(["profile:guided"], DEFAULT_PROMPT_PROFILE)

    assert len(candidates) == 1
    assert candidates[0].id == "profile:guided"
    assert candidates[0].api_prompt

    candidate = replace(
        candidates[0],
        execute_home_code_description="Candidate code capability. Extra detail.",
        get_history_description="Candidate history capability. Extra detail.",
        get_statistics_description="Candidate statistics capability. Extra detail.",
        get_logbook_description="Candidate logbook capability. Extra detail.",
        get_automation_description="Candidate automation capability. Extra detail.",
    )
    case = EvalCase(
        id="candidate-tool-descriptions",
        home="home_full",
        user_request="Describe the available tools.",
        required_actions=(RequiredAction("light", "turn_on", ("light.living",)),),
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
    _prompt = render_eval_system_prompt(runtime, tools)

    # Candidate descriptions flow through as the provider tool schemas, not as
    # duplicated first-sentence summaries in the system prompt.
    assert {tool.name: tool.description for tool in tools} == {
        "execute_home_code": candidate.execute_home_code_description,
        "get_history": candidate.get_history_description,
        "get_statistics": candidate.get_statistics_description,
        "get_logbook": candidate.get_logbook_description,
        "get_automation": candidate.get_automation_description,
    }

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


@pytest.mark.parametrize("profile_id", ["guided", "balanced", "frontier"])
def test_profiles_load_with_expected_size_order(profile_id: str) -> None:
    guided = load_candidates(["profile:guided"], DEFAULT_PROMPT_PROFILE)[0]
    balanced = load_candidates(["profile:balanced"], DEFAULT_PROMPT_PROFILE)[0]
    frontier = load_candidates(["profile:frontier"], DEFAULT_PROMPT_PROFILE)[0]
    candidate = load_candidates([f"profile:{profile_id}"], DEFAULT_PROMPT_PROFILE)[0]

    guided_api_chars, _guided_authored_chars = candidate_prompt_sizes(guided)
    balanced_api_chars, _balanced_authored_chars = candidate_prompt_sizes(balanced)
    frontier_api_chars, _frontier_authored_chars = candidate_prompt_sizes(frontier)
    assert candidate.id == f"profile:{profile_id}"
    assert candidate.api_prompt
    assert frontier_api_chars < balanced_api_chars < guided_api_chars


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
