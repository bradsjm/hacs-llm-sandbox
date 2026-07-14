"""Single-artifact persistence for native pydantic-evals reports."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

from pydantic import TypeAdapter
from pydantic_evals.reporting import EvaluationReport

from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.experiment import MatrixCellMeta, MatrixCellRef
from llm_sandbox_evals.schema import CaseOutcome, CaseTrace, EvalCase, PartialRunArtifact, variant_label
from llm_sandbox_evals.scoring import evaluate_case

type MatrixReport = EvaluationReport[MatrixCellRef, CaseTrace, MatrixCellMeta]
type JsonObject = dict[str, object]

_REPORT_ADAPTER: TypeAdapter[MatrixReport] = TypeAdapter(MatrixReport)
_PARTIAL_ADAPTER: TypeAdapter[PartialRunArtifact] = TypeAdapter(PartialRunArtifact)
_SCORING_VERSION = 7
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
    payload = cast(JsonObject, json.loads(_REPORT_ADAPTER.dump_json(report)))
    payload["scoring_version"] = _SCORING_VERSION
    report_content = _json_content(payload)
    error_content = _error_log_content(payload)
    _atomic_text_write(run_dir / "errors.log", error_content)
    _atomic_text_write(run_dir / "report.json", report_content)
    return run_dir


def load_report(run_dir: Path) -> MatrixReport:
    """Load a saved native pydantic-evals report artifact."""
    payload = json.loads((run_dir / "report.json").read_bytes())
    if not _contains_current_trace(payload):
        # Deliberately reject before Pydantic validation so legacy artifacts cannot
        # be silently reinterpreted by a future schema-compatible decoder.
        raise ValueError("legacy scoring artifact; rerun evaluation with scoring v7")
    return _REPORT_ADAPTER.validate_python(payload)


def rescore_trace(trace: CaseTrace) -> CaseOutcome:
    """Rescore a v7 trace using only its persisted required actions and ledger."""
    if trace.scoring_version != _SCORING_VERSION:
        raise ValueError("legacy scoring artifact; rerun evaluation with scoring v7")
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
    _atomic_text_write(path, _json_content(payload))


def _json_content(payload: object) -> str:
    """Render an indented JSON artifact before any filesystem mutation occurs."""
    return json.dumps(payload, indent=2) + "\n"


def _atomic_text_write(path: Path, content: str) -> None:
    """Replace one text artifact atomically so interruption never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            # State mutation point: retain the tempfile path before writing can fail.
            temporary = Path(handle.name)
            handle.write(content)
        temporary.replace(path)
    finally:
        # Branch boundary: leave either the prior/complete target or no temporary file after every failure mode.
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _error_log_content(payload: object) -> str:
    """Render one NDJSON execution-error record per incomplete report case."""
    lines = [json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in _error_log_records(payload)]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _error_log_records(payload: object) -> list[JsonObject]:
    """Project persisted report cases into stable execution-error log records."""
    report = _json_object_or_none(payload)
    cases = _json_list_or_empty(_field(report, "cases"))
    records: list[JsonObject] = []
    for report_case in cases:
        case_payload = _json_object_or_none(report_case)
        output = _json_object_or_none(_field(case_payload, "output"))
        outcome = _json_object_or_none(_field(output, "outcome"))
        provider_error = _field(output, "provider_error")
        execution_error = _json_object_or_none(_field(output, "execution_error"))
        if _field(outcome, "state") != "incomplete" or (provider_error is None and execution_error is None):
            continue
        diagnostics = _json_object_or_none(_field(output, "diagnostics"))
        reasoning_effort = _field(output, "reasoning_effort")
        model_id = _field(output, "model_id")
        records.append(
            {
                "classification": _field(diagnostics, "failure"),
                "case_id": _field(output, "case_id"),
                "candidate_id": _field(output, "candidate_id"),
                "model_id": model_id,
                "variant": _variant_or_none(model_id, reasoning_effort),
                "reasoning_effort": reasoning_effort,
                "temperature": _field(output, "temperature"),
                "conversation_id": _field(output, "conversation_id"),
                "user_request": _field(output, "user_request"),
                "exception_type": _field(execution_error, "exception_type"),
                "message": _field(execution_error, "message"),
                "status_code": _field(execution_error, "status_code"),
                "provider_code": _field(execution_error, "provider_code"),
                "provider_model": _field(execution_error, "provider_model"),
                "provider_detail": _field(execution_error, "provider_detail"),
                "detail": provider_error,
            }
        )
    return records


def _field(payload: JsonObject | None, key: str) -> object:
    """Read one JSON object field, returning null-equivalent when metadata is absent."""
    if payload is None:
        return None
    return payload.get(key)


def _json_object_or_none(payload: object) -> JsonObject | None:
    """Return one JSON object or null-equivalent for unexpected serialized shapes."""
    if not isinstance(payload, dict):
        return None
    return cast(JsonObject, payload)


def _json_list_or_empty(payload: object) -> list[object]:
    """Return one JSON list or an empty collection for unexpected serialized shapes."""
    if not isinstance(payload, list):
        return []
    return cast(list[object], payload)


def _variant_or_none(model_id: object, reasoning_effort: object) -> str | None:
    """Derive the existing display variant only when structured model identity exists."""
    if not isinstance(model_id, str):
        return None
    return variant_label(model_id, reasoning_effort if isinstance(reasoning_effort, str) else None)


def _contains_current_trace(payload: object) -> bool:
    """Check the serialized report envelope for the current self-contained trace shape."""
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
