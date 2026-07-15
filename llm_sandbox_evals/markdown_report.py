"""Deterministic Markdown rendering from the immutable report presentation model."""

from pathlib import Path
from tempfile import NamedTemporaryFile

from llm_sandbox_evals.presentation import ReportPresentationModel
from llm_sandbox_evals.statistics import canonical_cells, pair_aggregates, wilson_interval


def render_markdown(model: ReportPresentationModel) -> str:
    """Render a deterministic Markdown report from the immutable presentation model."""
    descriptor = model.descriptor
    scoring_version = model.cells[0].trace.scoring_version if model.cells else "—"
    lines = [
        "# LLM Sandbox Eval Report",
        "",
        f"- Run ID: `{_text(descriptor.get('run_id'))}`",
        f"- Scoring version: `{scoring_version}`",
        f"- Created at: `{_text(descriptor.get('created_at'))}`",
        "",
        "## Overall",
        "",
        "Canonical candidate/model leaderboard.",
        "",
        "| Candidate | Variant | Quality | Wilson 95% CI | Coverage | Scored | Total | Cost | Tokens |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        _row(
            (
                _escape(aggregate.candidate_id),
                _escape(aggregate.variant),
                _percentage(aggregate.counts.quality_rate),
                _interval(wilson_interval(aggregate.counts.correct, aggregate.counts.scored)),
                _percentage(aggregate.counts.coverage_rate),
                str(aggregate.counts.scored),
                str(aggregate.counts.total),
                _number(aggregate.total_cost),
                _number(aggregate.total_tokens),
            )
        )
        for aggregate in pair_aggregates(canonical_cells(model.cells))
    )

    lines.extend(
        (
            "",
            "## By Category",
            "",
            "| Candidate | Variant | Category | Quality | Coverage | Scored |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        )
    )
    lines.extend(
        _row(
            (
                _escape(aggregate.candidate_id),
                _escape(aggregate.variant),
                _escape(aggregate.category),
                _percentage(aggregate.counts.quality_rate),
                _percentage(aggregate.counts.coverage_rate),
                str(aggregate.counts.scored),
            )
        )
        for aggregate in model.category_aggregates
    )

    lines.extend(
        (
            "",
            "## Per Task",
            "",
            "| Case | Category | Candidate | Variant | Outcome | Scoring mode | Score reason |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        )
    )
    ordered_cells = sorted(
        model.cells,
        key=lambda value: (
            value.candidate_id,
            value.variant,
            value.category,
            value.case_id,
            value.request_variant_id,
        ),
    )
    lines.extend(
        _row(
            (
                _escape(cell.case_id),
                _escape(cell.category),
                _escape(cell.candidate_id),
                _escape(cell.variant),
                _escape(cell.trace.outcome.state),
                _escape(cell.trace.outcome.scoring_mode or "—"),
                _escape(cell.trace.outcome.score_reason or "—"),
            )
        )
        for cell in ordered_cells
    )

    lines.extend(
        (
            "",
            "## Method",
            "",
            f"Scoring version {scoring_version}. Quality = correct / scored; coverage = scored / total. "
            "Wilson 95% confidence intervals use scored canonical cells. Incomplete cells are excluded "
            "from quality. Canonical requests define the primary leaderboard; paraphrases remain distinct "
            "utterance-level cells, and their intervals are not independent task-level intervals.",
            "",
        )
    )

    # Branch boundary: advisory judge output is absent unless the presentation model records a request.
    if model.judge_requested:
        lines.extend(
            (
                "",
                "## Code Judge",
                "",
                "| Case | Request variant | Category | Candidate | Variant | Status | Score | Pass | Reason | Failure |",
                "| --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- |",
            )
        )
        lines.extend(
            _row(
                (
                    _escape(cell.case_id),
                    _escape(cell.request_variant_id),
                    _escape(cell.category),
                    _escape(cell.candidate_id),
                    _escape(cell.variant),
                    _escape(cell.judge.status),
                    _number(cell.judge.score),
                    "—" if cell.judge.passed is None else str(cell.judge.passed).lower(),
                    _escape(cell.judge.reason or "—"),
                    _escape((cell.judge.failure.error_type or "—") if cell.judge.failure is not None else "—"),
                )
            )
            for cell in ordered_cells
        )

    return "\n".join(lines)


def write_markdown(run_dir: Path, model: ReportPresentationModel) -> Path:
    """Atomically write report.md beside the persisted report artifacts."""
    path = run_dir / "report.md"
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            # State mutation point: retain the temporary path before writing can fail.
            temporary = Path(handle.name)
            handle.write(render_markdown(model))
        temporary.replace(path)
    finally:
        # Branch boundary: every failure leaves either the prior complete report or no temporary file.
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def _percentage(value: float | None) -> str:
    """Format one optional rate as a percentage."""
    return "—" if value is None else f"{value:.1%}"


def _row(values: tuple[str, ...]) -> str:
    """Render one standard Markdown table row."""
    return "| " + " | ".join(values) + " |"


def _interval(interval: tuple[float | None, float | None]) -> str:
    """Format an optional Wilson interval."""
    low, high = interval
    return "—" if low is None or high is None else f"[{low:.1%}, {high:.1%}]"


def _number(value: float | None) -> str:
    """Format an optional aggregate number deterministically."""
    return "—" if value is None else f"{value:g}"


def _text(value: object) -> str:
    """Render optional descriptor text without manufacturing metadata."""
    return "—" if value is None else str(value)


def _escape(value: str) -> str:
    """Escape Markdown table delimiters and line breaks in persisted text."""
    return value.replace("|", "\\|").replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
