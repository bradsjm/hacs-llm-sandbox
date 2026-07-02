from pathlib import Path

import pytest
from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from custom_components.llm_sandbox.llm_api.prompts import resolve_profile
from llm_sandbox_evals.cases import CASES
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.harness import _run_cases_for_pair, run_case
from llm_sandbox_evals.models import ModelResponseError
from llm_sandbox_evals.prompts import load_candidates
from llm_sandbox_evals.schema import AgentStep


class FailingAdapter:
    def __init__(self, *, detail: str = "provider rejected model") -> None:
        self.calls = 0
        self.detail = detail

    async def respond(
        self,
        model_id: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        *,
        force_text: bool = False,
    ) -> AgentStep:
        _ = (model_id, messages, tools, force_text)
        self.calls += 1
        raise ModelResponseError("provider rejected model", detail=self.detail)


async def test_run_case_records_model_error(tmp_path: Path) -> None:
    candidate = load_candidates(["baseline"], DEFAULT_PROMPT_PROFILE)[0]
    profile = resolve_profile(DEFAULT_PROMPT_PROFILE)
    config = EvalConfig(
        models=["bad-model"],
        candidates=[candidate.id],
        prompt_profile=DEFAULT_PROMPT_PROFILE,
        cases=None,
        homes=None,
        runs_dir=tmp_path,
    )

    trace = await run_case(candidate, "bad-model", CASES[0], FailingAdapter(), profile, config)

    assert trace.score == 0.0
    assert trace.raw_output == "provider rejected model"
    assert [(check.name, check.passed, check.required) for check in trace.checks] == [("model_error", False, True)]


async def test_pair_stops_calling_model_after_first_model_error(tmp_path: Path) -> None:
    candidate = load_candidates(["baseline"], DEFAULT_PROMPT_PROFILE)[0]
    profile = resolve_profile(DEFAULT_PROMPT_PROFILE)
    adapter = FailingAdapter()
    selected_cases = CASES[:3]
    config = EvalConfig(
        models=["bad-model"],
        candidates=[candidate.id],
        prompt_profile=DEFAULT_PROMPT_PROFILE,
        cases=None,
        homes=None,
        runs_dir=tmp_path,
        concurrency=1,
    )

    traces = await _run_cases_for_pair(candidate, "bad-model", selected_cases, adapter, config, profile)

    assert adapter.calls == 1
    assert [trace.case_id for trace in traces] == [case.id for case in selected_cases]
    assert [(trace.score, trace.checks[0].name) for trace in traces] == [
        (0.0, "model_error"),
        (0.0, "model_error"),
        (0.0, "model_error"),
    ]


async def test_pair_logs_provider_error_detail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    candidate = load_candidates(["baseline"], DEFAULT_PROMPT_PROFILE)[0]
    profile = resolve_profile(DEFAULT_PROMPT_PROFILE)
    adapter = FailingAdapter(detail="litellm.BadRequestError: provider rejected model\nstatus_code=400")
    config = EvalConfig(
        models=["bad-model"],
        candidates=[candidate.id],
        prompt_profile=DEFAULT_PROMPT_PROFILE,
        cases=None,
        homes=None,
        runs_dir=tmp_path,
        concurrency=1,
    )

    await _run_cases_for_pair(candidate, "bad-model", CASES[:2], adapter, config, profile)

    captured = capsys.readouterr()
    assert "model error for candidate=baseline model=bad-model case=" in captured.err
    assert "litellm.BadRequestError: provider rejected model" in captured.err
    assert "status_code=400" in captured.err
