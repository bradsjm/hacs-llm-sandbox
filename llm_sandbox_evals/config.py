"""Evaluation configuration: model matrix, candidate/case selection, run output directory."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Resolved eval configuration. API keys are read from the environment by the model adapter, never stored here."""

    models: list[str]
    candidates: list[str]
    cases: list[str] | None
    homes: list[str] | None
    runs_dir: Path
    concurrency: int = 5
    reasoning_effort: str | None = None
    target_model: str | None = None
    proposer_model: str | None = None
    target_reasoning_effort: str | None = None
    proposer_reasoning_effort: str | None = None
    breadth: int = 5
    depth: int = 2
    cross_eval_models: list[str] | None = None


def load_config() -> EvalConfig:
    """Return the default eval configuration.

    v1 keeps configuration in-process (KISS): real model ids are supplied via
    CLI flags; defaults to the offline stub adapter so the harness runs without
    API keys.
    """
    return EvalConfig(
        models=["stub"],
        candidates=["baseline"],
        cases=None,
        homes=None,
        runs_dir=Path("eval_data/runs"),
        concurrency=5,
    )
