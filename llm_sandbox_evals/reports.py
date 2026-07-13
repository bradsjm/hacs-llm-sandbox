"""Single-artifact persistence for native pydantic-evals reports."""

import json
from pathlib import Path

from pydantic import TypeAdapter
from pydantic_evals.reporting import EvaluationReport

from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.experiment import MatrixCellMeta, MatrixCellRef
from llm_sandbox_evals.schema import CaseOutcome, CaseTrace, EvalCase
from llm_sandbox_evals.scoring import evaluate_case

type MatrixReport = EvaluationReport[MatrixCellRef, CaseTrace, MatrixCellMeta]

_REPORT_ADAPTER: TypeAdapter[MatrixReport] = TypeAdapter(MatrixReport)
_SCORING_VERSION = 5
_CASE_TRACE_FIELDS = frozenset(
    {
        "case_id",
        "scoring_version",
        "candidate_id",
        "model_id",
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
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = json.loads(_REPORT_ADAPTER.dump_json(report))
    payload["scoring_version"] = _SCORING_VERSION
    (run_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return run_dir


def load_report(run_dir: Path) -> MatrixReport:
    """Load a saved native pydantic-evals report artifact."""
    payload = json.loads((run_dir / "report.json").read_bytes())
    if not _contains_v5_trace(payload):
        # Deliberately reject before Pydantic validation so legacy artifacts cannot
        # be silently reinterpreted by a future schema-compatible decoder.
        raise ValueError("legacy scoring artifact; rerun evaluation")
    return _REPORT_ADAPTER.validate_python(payload)


def rescore_trace(trace: CaseTrace) -> CaseOutcome:
    """Rescore a v5 trace using only its persisted required actions and ledger."""
    if trace.scoring_version != _SCORING_VERSION:
        raise ValueError("legacy scoring artifact; rerun evaluation")
    recorded_actions = trace.action_ledger.successful + trace.action_ledger.rejected
    case = EvalCase(
        id=trace.case_id,
        home="stored-trace",
        user_request=trace.user_request,
        required_actions=trace.required_actions,
    )
    outcome, _, _ = evaluate_case(case, recorded_actions)
    return outcome


def _contains_v5_trace(payload: object) -> bool:
    """Check the serialized report envelope for the self-contained v5 trace shape."""
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
