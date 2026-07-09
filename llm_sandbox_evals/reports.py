"""Single-artifact persistence for native pydantic-evals reports."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic_evals.reporting import EvaluationReport, ReportCase

from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.experiment import MatrixCellMeta, MatrixCellRef
from llm_sandbox_evals.schema import CaseTrace
from llm_sandbox_evals.scoring import is_incomplete

_INTEGRATION_MANIFEST = Path(__file__).resolve().parent.parent / "custom_components" / "llm_sandbox" / "manifest.json"


@dataclass(frozen=True, slots=True)
class ReportPayload:
    """Persisted report.json payload used by the no-model-calls report subcommand."""

    run_id: str
    created_at: str
    integration_version: str
    candidate_ids: list[str]
    model_ids: list[str]
    case_count: int
    incomplete_count: int
    analyses: list[dict[str, object]]
    cells: list[dict[str, object]]


def write_report_json(
    report: EvaluationReport[MatrixCellRef, CaseTrace, MatrixCellMeta],
    config: EvalConfig,
    *,
    run_id: str,
    created_at: str,
) -> Path:
    """Write the single report.json artifact and return its run directory."""
    run_dir = config.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    cells = [_cell_json(case) for case in report.cases]
    # Branch boundary: a cell is incomplete when it failed on a model_error gate
    # (provider outage / timeout). Such cells are excluded from mean scores but
    # kept visible in the trace for auditing.
    incomplete_count = sum(1 for case in report.cases if is_incomplete(case.output.checks))
    payload = ReportPayload(
        run_id=run_id,
        created_at=created_at,
        integration_version=_integration_version(),
        candidate_ids=_ordered_values(cells, "candidate_id"),
        model_ids=_ordered_values(cells, "model_id"),
        case_count=len(_ordered_values(cells, "case_id")),
        incomplete_count=incomplete_count,
        analyses=[analysis.model_dump(mode="json") for analysis in report.analyses],
        cells=cells,
    )
    (run_dir / "report.json").write_text(
        json.dumps(asdict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_dir


def load_report_payload(run_dir: Path) -> ReportPayload:
    """Load a saved native report payload."""
    decoded = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("report.json must contain a JSON object")
    return ReportPayload(
        run_id=_string_field(decoded, "run_id"),
        created_at=_string_field(decoded, "created_at"),
        integration_version=_string_field(decoded, "integration_version"),
        candidate_ids=_string_list(decoded.get("candidate_ids")),
        model_ids=_string_list(decoded.get("model_ids")),
        case_count=_int_field(decoded, "case_count"),
        incomplete_count=_int_field(decoded, "incomplete_count"),
        analyses=_dict_list(decoded.get("analyses")),
        cells=_dict_list(decoded.get("cells")),
    )


def render_report_summary(payload: ReportPayload) -> str:
    """Render saved analyses and cell scores as plain text without model calls."""
    lines = [
        f"Eval report {payload.run_id}",
        f"Created: {payload.created_at}",
        f"Cases: {payload.case_count}",
        f"Incomplete: {payload.incomplete_count}",
        "",
    ]
    for analysis in payload.analyses:
        title = str(analysis.get("title", "Analysis"))
        analysis_type = analysis.get("type")
        lines.extend([title, "-" * len(title)])
        if analysis_type == "scalar":
            lines.append(str(analysis.get("value", "")))
        elif analysis_type == "table":
            raw_columns = analysis.get("columns", [])
            columns = [str(column) for column in raw_columns] if isinstance(raw_columns, list) else []
            rows = analysis.get("rows", [])
            lines.append("\t".join(columns))
            if isinstance(rows, list):
                lines.extend("\t".join(str(value) for value in row) for row in rows if isinstance(row, list))
        lines.append("")
    lines.append("Cells")
    lines.append("-----")
    for cell in payload.cells:
        score = cell.get("score", 0.0)
        score_float = float(score) if isinstance(score, int | float) else 0.0
        trace = cell.get("trace")
        error = trace.get("error") if isinstance(trace, dict) else None
        # Branch boundary: persisted error details live inside trace.error; append
        # them only for failed cells so successful summaries stay compact.
        error_text = f" error={error}" if isinstance(error, str) and error else ""
        lines.append(
            f"{cell.get('candidate_id')}/{cell.get('model_id')}/{cell.get('case_id')}: "
            f"score={score_float:.3f} tool_calls={cell.get('tool_calls')}{error_text}"
        )
    return "\n".join(lines) + "\n"


def _integration_version() -> str:
    """Read the integration version from the checked-in Home Assistant manifest."""
    decoded = json.loads(_INTEGRATION_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("manifest.json must contain a JSON object")
    version = decoded.get("version")
    if not isinstance(version, str):
        raise ValueError("manifest.json field 'version' must be a string")
    return version


def _cell_json(report_case: ReportCase[MatrixCellRef, CaseTrace, MatrixCellMeta]) -> dict[str, object]:
    trace: CaseTrace = report_case.output
    metadata = report_case.metadata or {}
    score = report_case.scores.get("score")
    return {
        "case_id": str(metadata["case_id"]),
        "category": str(metadata["category"]),
        "candidate_id": str(metadata["candidate_id"]),
        "model_id": str(metadata["model_id"]),
        "score": 0.0 if score is None else float(score.value),
        "tool_calls": trace.tool_call_count,
        "checks": [asdict(check) for check in trace.checks],
        "trace": _trace_json(trace),
    }


def _trace_json(trace: CaseTrace) -> dict[str, object]:
    """Build one full per-cell trace artifact inside report.json."""
    return {
        "case_id": trace.case_id,
        "category": trace.category,
        "candidate_id": trace.candidate_id,
        "model_id": trace.model_id,
        "score": trace.score,
        "output": trace.output,
        "tool_call_count": trace.tool_call_count,
        "recorded_actions": list(trace.recorded_actions),
        "tool_events": [
            {"tool_name": event.tool_name, "args": event.args, "output": event.output} for event in trace.tool_events
        ],
        "checks": [asdict(check) for check in trace.checks],
        "error": trace.error,
    }


def _ordered_values(rows: list[dict[str, object]], key: str) -> list[str]:
    values: list[str] = []
    for row in rows:
        value = str(row[key])
        if value not in values:
            values.append(value)
    return values


def _string_field(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"report.json field {key!r} must be a string")
    return value


def _int_field(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise ValueError(f"report.json field {key!r} must be an integer")
    return value


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("report.json field must be a list")
    return [item for item in value if isinstance(item, str)]


def _dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("report.json field must be a list")
    return [dict(item) for item in value if isinstance(item, dict)]
