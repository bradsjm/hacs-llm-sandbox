import json
from pathlib import Path

import pytest
from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE, TOOL_GET_HISTORY
from custom_components.llm_sandbox.llm_api.tools._aggregates import AGGREGATORS
from llm_sandbox_evals.optimize_dspy import size_penalized_utility
from llm_sandbox_evals.prompts import baseline_candidate, candidate_prompt_sizes, function_schemas, load_candidates
from llm_sandbox_evals.reports import load_run_json, render_leaderboard_from_scores
from llm_sandbox_evals.schema import CandidateModelScore, PromptCandidate


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


def test_candidate_prompt_sizes_counts_api_and_authored_prompt_text() -> None:
    candidate = PromptCandidate(
        id="t",
        api_prompt="abc",
        execute_home_code_description="d1",
        get_history_description="d2",
        get_statistics_description="d3",
        get_logbook_description="d4",
    )

    api_prompt_chars, authored_prompt_chars = candidate_prompt_sizes(candidate)

    assert api_prompt_chars == 3
    assert authored_prompt_chars == 11


def test_get_history_function_schema_exposes_aggregate_filters() -> None:
    schemas = function_schemas(baseline_candidate())
    history_schema = next(schema for schema in schemas if schema["function"]["name"] == TOOL_GET_HISTORY)

    parameters = history_schema["function"]["parameters"]
    assert parameters["additionalProperties"] is False
    properties = parameters["properties"]
    assert properties["aggregate"] == {"type": "string", "enum": list(AGGREGATORS)}
    assert properties["from_state"] == {"type": "string"}
    assert properties["to_state"] == {"type": "string"}


@pytest.mark.parametrize(
    ("score", "ratio", "penalty", "expected", "expected_less_than_score"),
    [
        pytest.param(0.9, 1.0, 0.02, 0.9, False, id="baseline-size-no-penalty"),
        pytest.param(0.9, 0.5, 0.02, 0.9, False, id="smaller-size-no-reward"),
        pytest.param(0.9, 2.0, 0.02, 0.88, True, id="larger-size-linear-penalty"),
    ],
)
def test_size_penalized_utility(
    score: float,
    ratio: float,
    penalty: float,
    expected: float,
    expected_less_than_score: bool,
) -> None:
    utility = size_penalized_utility(score, ratio, penalty)

    assert utility == pytest.approx(expected)
    assert (utility < score) is expected_less_than_score


def test_leaderboard_ranks_smaller_api_prompt_first_for_equal_mean() -> None:
    scores = [
        CandidateModelScore(
            candidate_id="big",
            model_id="stub",
            mean=0.5,
            mean_turns=1.0,
            per_category={"intent": 0.5},
            case_scores={"case": 0.5},
            api_prompt_chars=1000,
            prompt_chars=2000,
        ),
        CandidateModelScore(
            candidate_id="small",
            model_id="stub",
            mean=0.5,
            mean_turns=1.0,
            per_category={"intent": 0.5},
            case_scores={"case": 0.5},
            api_prompt_chars=500,
            prompt_chars=1000,
        ),
    ]

    out = render_leaderboard_from_scores(
        scores=scores,
        run_id="t",
        created_at="t",
        case_count=1,
        candidate_ids=["big", "small"],
        model_ids=["stub"],
    )

    assert out.index("small") < out.index("big")
    assert "PromptChars" in out
    assert "SizeRatio" in out


@pytest.mark.parametrize(
    ("score_fields", "expected_api_prompt_chars", "expected_prompt_chars"),
    [
        pytest.param(
            {"api_prompt_chars": 7271, "prompt_chars": 10156},
            7271,
            10156,
            id="with-sizes",
        ),
        pytest.param({}, 0, 0, id="legacy-defaults"),
    ],
)
def test_load_run_json_preserves_prompt_sizes(
    tmp_path: Path,
    score_fields: dict[str, object],
    expected_api_prompt_chars: int,
    expected_prompt_chars: int,
) -> None:
    run_json = tmp_path / "run.json"
    run_json.write_text(
        json.dumps(
            {
                "run_id": "run",
                "created_at": "2026-07-02T00:00:00+00:00",
                "candidate_ids": ["candidate"],
                "model_ids": ["stub"],
                "case_count": 1,
                "scores": [
                    {
                        "candidate_id": "candidate",
                        "model_id": "stub",
                        "mean": 0.75,
                        "mean_turns": 1.0,
                        "per_category": {"intent": 0.75},
                        "case_scores": {"case": 0.75},
                        **score_fields,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    _run_id, _created_at, _case_count, _candidate_ids, _model_ids, scores = load_run_json(run_json)

    assert scores[0].api_prompt_chars == expected_api_prompt_chars
    assert scores[0].prompt_chars == expected_prompt_chars
