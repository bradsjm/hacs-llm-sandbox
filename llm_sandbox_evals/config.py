"""Evaluation configuration: model matrix, candidate/case selection, run output directory."""

from dataclasses import dataclass
from pathlib import Path

from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Resolved eval configuration. API keys are read from the environment by the model adapter, never stored here."""

    models: list[str]
    candidates: list[str]
    prompt_profile: str
    cases: list[str] | None
    homes: list[str] | None
    runs_dir: Path
    concurrency: int = 5
    max_tool_calls: int = 8
    model_timeout: float = 75.0
    reasoning_effort: str | None = None
    target_model: str | None = None
    proposer_model: str | None = None
    target_reasoning_effort: str | None = None
    proposer_reasoning_effort: str | None = None
    breadth: int = 5
    depth: int = 2
    cross_eval_models: list[str] | None = None
    length_penalty: float = 0.02  # Penalizes api_prompt growth in COPRO; smaller = weaker tie-break.


def load_config() -> EvalConfig:
    """Return the default eval configuration.

    v1 keeps configuration in-process (KISS): real model ids are supplied via
    CLI flags; defaults to the offline stub adapter so the harness runs without
    API keys.
    """
    return EvalConfig(
        models=["stub"],
        candidates=["baseline"],
        prompt_profile=DEFAULT_PROMPT_PROFILE,
        cases=None,
        homes=None,
        runs_dir=Path("eval_data/runs"),
        concurrency=5,
    )
