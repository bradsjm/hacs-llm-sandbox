"""Single-artifact persistence for native pydantic-evals reports."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import TypeAdapter
from pydantic_evals.reporting import EvaluationReport

from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.experiment import MatrixCellMeta, MatrixCellRef
from llm_sandbox_evals.schema import CaseOutcome, CaseTrace, EvalCase, PartialRunArtifact
from llm_sandbox_evals.scoring import evaluate_case

type MatrixReport = EvaluationReport[MatrixCellRef, CaseTrace, MatrixCellMeta]

_REPORT_ADAPTER: TypeAdapter[MatrixReport] = TypeAdapter(MatrixReport)
_PARTIAL_ADAPTER: TypeAdapter[PartialRunArtifact] = TypeAdapter(PartialRunArtifact)
_SCORING_VERSION = 6
_CASE_TRACE_FIELDS = frozenset(
    {
        "case_id",
        "scoring_version",
        "candidate_id",
        "model_id",
        "reasoning_effort",
        "temperature",
        "answer",
        "required_actions",
        "outcome",
        "action_result",
        "action_ledger",
        "tool_events",
        "diagnostics",
        "provider_error",
        "user_request",
        "conversation_id",
    }
)


def write_report_json(
    report: MatrixReport,
    config: EvalConfig,
    *,
    run_id: str,
) -> Path:
    """Write the native pydantic-evals report artifact and return its run directory."""
    run_dir = config.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(_REPORT_ADAPTER.dump_json(report))
    payload["scoring_version"] = _SCORING_VERSION
    _atomic_json_write(run_dir / "report.json", payload)
    return run_dir


def load_report(run_dir: Path) -> MatrixReport:
    """Load a saved native pydantic-evals report artifact."""
    payload = json.loads((run_dir / "report.json").read_bytes())
    if not _contains_v6_trace(payload):
        # Deliberately reject before Pydantic validation so legacy artifacts cannot
        # be silently reinterpreted by a future schema-compatible decoder.
        raise ValueError("legacy scoring-v6 artifact; rerun evaluation")
    return _REPORT_ADAPTER.validate_python(payload)


def rescore_trace(trace: CaseTrace) -> CaseOutcome:
    """Rescore a v6 trace using only its persisted required actions and ledger."""
    if trace.scoring_version != _SCORING_VERSION:
        raise ValueError("legacy scoring-v6 artifact; rerun evaluation")
    recorded_actions = trace.action_ledger.successful + trace.action_ledger.rejected
    case = EvalCase(
        id=trace.case_id,
        home="stored-trace",
        user_request=trace.user_request,
        required_actions=trace.required_actions,
    )
    outcome, _, _ = evaluate_case(case, recorded_actions)
    return outcome


def write_partial_artifact(path: Path, artifact: PartialRunArtifact) -> Path:
    """Atomically write a typed partial-run journal that is explicitly not a report."""
    _atomic_json_write(path, json.loads(_PARTIAL_ADAPTER.dump_json(artifact)))
    return path


def load_partial_artifact(path: Path) -> PartialRunArtifact:
    """Load one typed partial-run journal without treating it as an EvaluationReport."""
    return _PARTIAL_ADAPTER.validate_json(path.read_bytes())


def write_manifest(path: Path, payload: dict[str, object]) -> Path:
    """Atomically write the small lifecycle manifest for a run directory."""
    _atomic_json_write(path, payload)
    return path


def _atomic_json_write(path: Path, payload: object) -> None:
    """Replace one JSON artifact atomically so interruption never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            # State mutation point: retain the tempfile path before serialization or writing can fail.
            temporary = Path(handle.name)
            handle.write(json.dumps(payload, indent=2) + "\n")
        temporary.replace(path)
    finally:
        # Branch boundary: leave either the prior/complete target or no temporary file after every failure mode.
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _contains_v6_trace(payload: object) -> bool:
    """Check the serialized report envelope for the self-contained v6 trace shape."""
    if (
        not isinstance(payload, dict)
        or payload.get("scoring_version") != _SCORING_VERSION
        or not isinstance(payload.get("cases"), list)
        or not payload["cases"]
    ):
        return False
    for report_case in payload["cases"]:
        if not isinstance(report_case, dict) or not isinstance(report_case.get("output"), dict):
            return False
        output = report_case["output"]
        if not _CASE_TRACE_FIELDS.issubset(output) or output.get("scoring_version") != _SCORING_VERSION:
            return False
        outcome = output["outcome"]
        if not isinstance(outcome, dict) or outcome.get("state") not in {"correct", "incorrect", "incomplete"}:
            return False
    return True
