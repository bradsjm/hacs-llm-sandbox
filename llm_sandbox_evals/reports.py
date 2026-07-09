"""Single-artifact persistence for native pydantic-evals reports."""

from pathlib import Path

from pydantic import TypeAdapter
from pydantic_evals.reporting import EvaluationReport

from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.experiment import MatrixCellMeta, MatrixCellRef
from llm_sandbox_evals.schema import CaseTrace

type MatrixReport = EvaluationReport[MatrixCellRef, CaseTrace, MatrixCellMeta]

_REPORT_ADAPTER: TypeAdapter[MatrixReport] = TypeAdapter(MatrixReport)


def write_report_json(
    report: MatrixReport,
    config: EvalConfig,
    *,
    run_id: str,
) -> Path:
    """Write the native pydantic-evals report artifact and return its run directory."""
    run_dir = config.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "report.json").write_bytes(_REPORT_ADAPTER.dump_json(report, indent=2) + b"\n")
    return run_dir


def load_report(run_dir: Path) -> MatrixReport:
    """Load a saved native pydantic-evals report artifact."""
    return _REPORT_ADAPTER.validate_json((run_dir / "report.json").read_bytes())
