"""DSPy COPRO prompt optimization for the dev-only eval harness."""

# mypy: disable-error-code="misc"

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import dspy
from custom_components.llm_sandbox.llm_api.prompts import PromptProfile, resolve_profile

from llm_sandbox_evals import experiment, prompts, reports
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.harness import _select_cases, run_case
from llm_sandbox_evals.optimize_helpers import _to_pydantic_ai_model_id, size_penalized_utility
from llm_sandbox_evals.schema import EvalCase, PromptCandidate


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
    baseline_prompt_chars: int
    optimized_prompt_chars: int
    size_ratio: float
    optimized_full_mean: float


def run_optimize(config: EvalConfig) -> OptimizerResult:
    """Run DSPy COPRO against one target model and export the optimized prompt candidate."""
    target = config.target_model
    # Branch boundary: DSPy optimization requires a real provider-backed LM, not the offline stub adapter.
    if target is None or target == "stub":
        raise ValueError("optimize requires a real --target-model; 'stub' is not supported")
    # Branch boundary: DSPy COPRO rejects breadth <= 1 ("Breadth must be greater than 1"); surface it clearly.
    if config.breadth < 2:
        raise ValueError("optimize requires --breadth >= 2 (DSPy COPRO needs breadth > 1)")

    proposer = config.proposer_model or target
    profile = resolve_profile(config.prompt_profile)
    baseline = prompts.baseline_candidate(config.prompt_profile)
    target_lm = dspy.LM(
        model=target,
        **_dspy_reasoning_kwargs(temperature=0.0, reasoning_effort=config.target_reasoning_effort),
    )
    # Configure the global default LM as a defensive default; actual scoring runs through run_case's Pydantic AI agent.
    dspy.configure(lm=target_lm)
    prompt_lm = dspy.LM(
        model=proposer,
        **_dspy_reasoning_kwargs(temperature=1.0, reasoning_effort=config.proposer_reasoning_effort),
    )

    selected_cases = _select_cases(config.cases, config.homes)
    trainset = [
        dspy.Example(context=case.user_request, case=case).with_inputs("context", "case") for case in selected_cases
    ]
    pydantic_target = _to_pydantic_ai_model_id(target)
    student = _PromptInstructionStudent(baseline, pydantic_target, profile, config)
    copro = dspy.COPRO(
        prompt_model=prompt_lm,
        metric=_make_metric(),
        breadth=config.breadth,
        depth=config.depth,
    )
    # eval_kwargs forwards to dspy.Evaluate for candidate scoring. num_threads=1
    # keeps the sync metric's asyncio.run loop single. display_progress=True so
    # the long optimization reports its own activity.
    compiled = copro.compile(student, trainset=trainset, eval_kwargs={"num_threads": 1, "display_progress": True})
    optimized_instruction = str(compiled.predictor.signature.instructions)
    optimized_candidate = replace(baseline, id="optimized", api_prompt=optimized_instruction)

    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    created_at = datetime.now(UTC).isoformat()
    run_dir = config.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = run_dir / "optimized_candidate.json"
    save_candidate(optimized_candidate, candidate_path)
    (run_dir / "optimized_prompt.md").write_text(optimized_instruction, encoding="utf-8")

    baseline_mean = _candidate_mean(
        EvalConfig(
            models=[pydantic_target],
            candidates=["baseline"],
            prompt_profile=config.prompt_profile,
            cases=config.cases,
            homes=config.homes,
            runs_dir=config.runs_dir,
            max_tool_calls=config.max_tool_calls,
            reasoning_effort=config.target_reasoning_effort,
        )
    )
    optimized_mean = _candidate_mean(
        EvalConfig(
            models=[pydantic_target],
            candidates=[f"optimized:{candidate_path}"],
            prompt_profile=config.prompt_profile,
            cases=config.cases,
            homes=config.homes,
            runs_dir=config.runs_dir,
            max_tool_calls=config.max_tool_calls,
            reasoning_effort=config.target_reasoning_effort,
        )
    )
    # Branch boundary: this adds one full case-suite pass on the target model, increasing paid model-call cost.
    optimized_full_mean = _candidate_mean(
        EvalConfig(
            models=[pydantic_target],
            candidates=[f"optimized:{candidate_path}"],
            prompt_profile=config.prompt_profile,
            cases=None,
            homes=config.homes,
            runs_dir=config.runs_dir,
            max_tool_calls=config.max_tool_calls,
            reasoning_effort=config.target_reasoning_effort,
        )
    )

    cross_eval_run_dir: Path | None = None
    # Branch boundary: cross-evaluation is optional because it can multiply paid model calls.
    if config.cross_eval_models:
        cross_config = EvalConfig(
            models=[_to_pydantic_ai_model_id(model) for model in config.cross_eval_models],
            candidates=["baseline", f"optimized:{candidate_path}"],
            prompt_profile=config.prompt_profile,
            cases=config.cases,
            homes=config.homes,
            runs_dir=config.runs_dir,
            max_tool_calls=config.max_tool_calls,
            reasoning_effort=config.reasoning_effort,
        )
        cross_report = asyncio.run(experiment.run_matrix(cross_config))
        cross_eval_run_dir = reports.write_report_json(
            cross_report,
            cross_config,
            run_id=datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f"),
        )

    baseline_prompt_chars, baseline_authored = prompts.candidate_prompt_sizes(baseline)
    _ = baseline_authored
    optimized_prompt_chars, _ = prompts.candidate_prompt_sizes(optimized_candidate)
    size_ratio = optimized_prompt_chars / max(1, baseline_prompt_chars)

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
        baseline_prompt_chars=baseline_prompt_chars,
        optimized_prompt_chars=optimized_prompt_chars,
        size_ratio=size_ratio,
        optimized_full_mean=optimized_full_mean,
    )


class _PromptInstructionStudent(dspy.Module):
    """DSPy program whose optimized predictor instruction becomes ``PromptCandidate.api_prompt``."""

    def __init__(
        self,
        baseline: PromptCandidate,
        target_model: str,
        profile: PromptProfile,
        config: EvalConfig,
    ) -> None:
        super().__init__()
        self.baseline = baseline
        self.target_model = target_model
        self.profile = profile
        self.config = config
        self.predictor = dspy.Predict(dspy.Signature("context, case -> score", instructions=baseline.api_prompt))

    def forward(self, context: str, case: EvalCase) -> object:
        """Score the current COPRO-mutated instruction through the real multi-turn runner."""
        _ = context
        instruction = str(self.predictor.signature.instructions)
        candidate = replace(self.baseline, id="optimized", api_prompt=instruction)
        scoring_config = replace(self.config, reasoning_effort=self.config.target_reasoning_effort)
        trace = asyncio.run(
            run_case(
                candidate,
                self.target_model,
                case,
                scoring_config,
                profile=self.profile,
            )
        )
        ratio = len(instruction) / max(1, len(self.baseline.api_prompt))
        # Penalize COPRO's internal candidate selection only; reported means stay raw quality scores.
        return dspy.Prediction(score=size_penalized_utility(trace.score, ratio, self.config.length_penalty))


def _make_metric() -> Callable[..., float]:
    """Build the sync COPRO metric; the program already ran the real multi-turn loop."""

    def metric(example: Any, prediction: Any, trace: Any = None) -> float:  # noqa: ANN401, ARG001
        try:
            score = getattr(prediction, "score", 0.0)
            return float(score) if isinstance(score, int | float) else 0.0
        except Exception:  # noqa: BLE001 - optimizer metrics must not abort COPRO.
            return 0.0

    return metric


def _dspy_reasoning_kwargs(*, temperature: float, reasoning_effort: str | None) -> dict[str, object]:
    """Map DSPy decoding intent onto its litellm-backed kwargs."""
    # Branch boundary: DSPy's dspy.LM is litellm-backed, so this remains local to the optimize path.
    if reasoning_effort is None:
        return {"temperature": temperature}
    kwargs: dict[str, object] = {"extra_body": {"reasoning_effort": reasoning_effort}}
    # Branch boundary: explicit no-reasoning keeps deterministic decoding.
    if reasoning_effort == "none":
        kwargs["temperature"] = temperature
    return kwargs


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
        report = asyncio.run(experiment.run_matrix(config))
    except Exception:  # noqa: BLE001 - summary scoring should not hide an exported optimizer artifact.
        return 0.0
    return experiment.overall_mean(report)
