"""DSPy COPRO prompt optimization for the dev-only eval harness."""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import dspy

from llm_sandbox_evals import prompts, reports, tools
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.harness import _select_cases, run_matrix
from llm_sandbox_evals.homes import get_home
from llm_sandbox_evals.models import litellm_reasoning_kwargs, parse_tool_call
from llm_sandbox_evals.schema import PromptCandidate
from llm_sandbox_evals.scoring import check_case, mean_score, score_case


@dataclass(frozen=True, slots=True)
class OptimizerResult:
    """Result metadata for one DSPy optimizer run."""

    run_id: str
    created_at: str
    target_model: str
    proposer_model: str
    baseline_mean: float
    optimized_mean: float
    optimized_candidate: PromptCandidate
    candidate_path: Path
    cross_eval_run_dir: Path | None


def run_optimize(config: EvalConfig) -> OptimizerResult:
    """Run DSPy COPRO against one target model and export the optimized prompt candidate."""
    target = config.target_model
    # Branch boundary: DSPy optimization requires a real provider-backed LM, not the offline stub adapter.
    if target is None or target == "stub":
        raise ValueError("optimize requires a real --target-model; 'stub' is not supported")

    proposer = config.proposer_model or target
    baseline = prompts.baseline_candidate()
    # Per-role reasoning: COPRO scores candidate instructions against the target
    # LM and proposes rewrites with the proposer LM. Reasoning is forwarded to
    # each dspy.LM via the same provider contract the eval adapter uses.
    target_lm = dspy.LM(
        model=target,
        **litellm_reasoning_kwargs(temperature=0.0, reasoning_effort=config.target_reasoning_effort),
    )
    dspy.configure(lm=target_lm)
    prompt_lm = dspy.LM(
        model=proposer,
        **litellm_reasoning_kwargs(temperature=1.0, reasoning_effort=config.proposer_reasoning_effort),
    )

    trainset = []
    for case in _select_cases(config.cases, config.homes):
        snapshot = get_home(case.home).snapshot()
        context = prompts.render_context(baseline, case, snapshot)
        trainset.append(dspy.Example(context=context, _case=case, _snapshot=snapshot).with_inputs("context"))

    sig = dspy.Signature("context -> tool_call_json", instructions=baseline.api_prompt)
    student = dspy.Predict(sig)
    copro = dspy.COPRO(prompt_model=prompt_lm, metric=_make_metric(), breadth=config.breadth, depth=config.depth)
    # eval_kwargs forwards to dspy.Evaluate for candidate scoring. num_threads=1
    # keeps the sync metric's asyncio.run loop single (the executor is async).
    # display_progress=True so the long optimization reports its own activity.
    compiled = copro.compile(student, trainset=trainset, eval_kwargs={"num_threads": 1, "display_progress": True})
    optimized_instruction = str(compiled.signature.instructions)
    optimized_candidate = PromptCandidate(
        id="optimized",
        api_prompt=optimized_instruction,
        execute_home_code_description=baseline.execute_home_code_description,
        get_history_description=baseline.get_history_description,
        get_statistics_description=baseline.get_statistics_description,
        get_logbook_description=baseline.get_logbook_description,
    )

    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    created_at = datetime.now(UTC).isoformat()
    run_dir = config.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = run_dir / "optimized_candidate.json"
    save_candidate(optimized_candidate, candidate_path)
    (run_dir / "optimized_prompt.md").write_text(optimized_instruction, encoding="utf-8")

    baseline_mean = _candidate_mean(
        EvalConfig(
            models=[target],
            candidates=["baseline"],
            cases=config.cases,
            homes=config.homes,
            runs_dir=config.runs_dir,
            reasoning_effort=config.target_reasoning_effort,
        )
    )
    optimized_mean = _candidate_mean(
        EvalConfig(
            models=[target],
            candidates=[f"optimized:{candidate_path}"],
            cases=config.cases,
            homes=config.homes,
            runs_dir=config.runs_dir,
            reasoning_effort=config.target_reasoning_effort,
        )
    )

    cross_eval_run_dir: Path | None = None
    # Branch boundary: cross-evaluation is optional because it can multiply paid model calls.
    if config.cross_eval_models:
        cross_result = asyncio.run(
            run_matrix(
                EvalConfig(
                    models=config.cross_eval_models,
                    candidates=["baseline", f"optimized:{candidate_path}"],
                    cases=config.cases,
                    homes=config.homes,
                    runs_dir=config.runs_dir,
                    reasoning_effort=config.reasoning_effort,
                )
            )
        )
        cross_eval_run_dir = reports.write_run(cross_result, config.runs_dir)

    return OptimizerResult(
        run_id=run_id,
        created_at=created_at,
        target_model=target,
        proposer_model=proposer,
        baseline_mean=baseline_mean,
        optimized_mean=optimized_mean,
        optimized_candidate=optimized_candidate,
        candidate_path=candidate_path,
        cross_eval_run_dir=cross_eval_run_dir,
    )


def _make_metric() -> Callable[..., float]:
    """Build the sync COPRO metric backed by the real eval tool runner and scorer."""

    def metric(example: Any, prediction: Any, trace: Any = None) -> float:  # noqa: ANN401, ARG001
        try:
            tool_call = parse_tool_call(getattr(prediction, "tool_call_json", "") or "")
            case = example._case
            snapshot = example._snapshot
            outcome = asyncio.run(tools.run_tool(tool_call, case, snapshot))
            checks = check_case(case, tool_call, outcome, snapshot)
            return score_case(checks)
        except Exception:  # noqa: BLE001 - optimizer metrics must not abort COPRO.
            return 0.0

    return metric


def save_candidate(candidate: PromptCandidate, path: Path) -> None:
    """Write an optimized prompt candidate JSON artifact."""
    data = {
        "id": candidate.id,
        "api_prompt": candidate.api_prompt,
        "execute_home_code_description": candidate.execute_home_code_description,
        "get_history_description": candidate.get_history_description,
        "get_statistics_description": candidate.get_statistics_description,
        "get_logbook_description": candidate.get_logbook_description,
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _candidate_mean(config: EvalConfig) -> float:
    """Run one candidate/model matrix and return its mean, isolating evaluation failures."""
    try:
        result = asyncio.run(run_matrix(config))
    except Exception:  # noqa: BLE001 - summary scoring should not hide an exported optimizer artifact.
        return 0.0
    if not result.scores:
        return 0.0
    if len(result.scores) == 1:
        return result.scores[0].mean
    return mean_score([score.mean for score in result.scores])
