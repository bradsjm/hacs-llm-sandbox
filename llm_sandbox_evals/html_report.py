"""Small self-contained HTML renderer for action-only eval reports."""

from datetime import UTC, datetime
from html import escape
import json
from pathlib import Path


def write_html(run_dir: Path) -> Path:
    """Render report.json in run_dir into report.html and return its path."""
    report_json = run_dir / "report.json"
    report = json.loads(report_json.read_text(encoding="utf-8"))
    run_id = run_dir.name
    try:
        created_at = datetime.strptime(run_id, "%Y%m%d-%H%M%S-%f").replace(tzinfo=UTC).isoformat()
    except ValueError:
        created_at = datetime.fromtimestamp(report_json.stat().st_mtime, tz=UTC).isoformat()
    report_html = run_dir / "report.html"
    report_html.write_text(render_html(report, run_id=run_id, created_at=created_at), encoding="utf-8")
    return report_html


def render_html(report: dict[str, object], *, run_id: str | None = None, created_at: str | None = None) -> str:
    """Return an accessible static report focused on action outcomes and ledgers."""
    raw_cases = report.get("cases")
    cases = [case for case in raw_cases if isinstance(case, dict)] if isinstance(raw_cases, list) else []
    rows = "".join(_case_row(case) for case in cases)
    correct = sum(_outcome(case) == "correct" for case in cases)
    incomplete = sum(_outcome(case) == "incomplete" for case in cases)
    completed = len(cases) - incomplete
    rate = correct / completed if completed else 0.0
    title = f"Action eval report {run_id or ''}".strip()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="color-scheme" content="light dark">
  <title>{escape(title)}</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; line-height: 1.5; }}
    body {{ margin: 0; background: Canvas; color: CanvasText; }}
    main {{ width: min(76rem, calc(100% - 2rem)); margin: 2rem auto; }}
    h1 {{ font-size: 1.75rem; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-block: 1.5rem; }}
    .summary div {{ border: 1px solid GrayText; border-radius: .5rem; padding: .75rem 1rem; }}
    .summary strong {{ display: block; font-size: 1.35rem; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; }}
    caption {{ text-align: start; font-weight: 700; margin-block-end: .5rem; }}
    th, td {{ border-block-end: 1px solid GrayText; padding: .65rem; text-align: start; vertical-align: top; }}
    .correct {{ color: #16803a; font-weight: 700; }}
    .incorrect, .incomplete {{ color: #b42318; font-weight: 700; }}
    details {{ min-width: 18rem; }}
    .assessment {{ margin-block: .75rem; }}
    .assessment th, .assessment td {{ font-size: .9rem; }}
    .match {{ color: #16803a; font-weight: 700; }}
    .mismatch {{ color: #b42318; font-weight: 700; }}
    summary {{ cursor: pointer; }}
    pre {{ max-width: 60rem; max-height: 30rem; overflow: auto; white-space: pre-wrap; }}
    :focus-visible {{ outline: 3px solid #0b7285; outline-offset: 2px; }}
  </style>
</head>
<body>
  <main id="content" tabindex="-1">
    <h1>{escape(title)}</h1>
    <p>Created: {escape(created_at or "unknown")}</p>
    <section aria-labelledby="summary-heading">
      <h2 id="summary-heading">Overall model comparison</h2>
      <div class="summary">
        <div><strong>{rate:.3f}</strong>Correct rate</div>
        <div><strong>{correct}</strong>Correct</div>
        <div><strong>{completed - correct}</strong>Incorrect</div>
        <div><strong>{incomplete}</strong>Incomplete</div>
      </div>
    </section>
    <section aria-labelledby="results-heading">
      <h2 id="results-heading">Action results</h2>
      <div class="table-wrap">
        <table>
          <caption>Successful ledger scoring by case, candidate, and model</caption>
          <thead><tr><th scope="col">Case</th><th scope="col">Candidate</th><th scope="col">Model</th><th scope="col">Outcome</th><th scope="col">Details</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>
"""


def _case_row(case: dict[str, object]) -> str:
    raw_output = case.get("output")
    output: dict[str, object] = raw_output if isinstance(raw_output, dict) else {}
    raw_inputs = case.get("inputs")
    inputs: dict[str, object] = raw_inputs if isinstance(raw_inputs, dict) else {}
    outcome = _outcome(case)
    assessment = _action_assessment(output.get("action_result"))
    detail = json.dumps(
        {
            "request": output.get("user_request"),
            "answer": output.get("answer"),
            "expected_actions": output.get("expected_actions"),
            "action_result": output.get("action_result"),
            "action_ledger": output.get("action_ledger"),
            "tool_events": output.get("tool_events"),
            "diagnostics": output.get("diagnostics"),
        },
        indent=2,
    )
    return (
        "<tr>"
        f"<td>{escape(str(inputs.get('case_id', case.get('name', ''))))}</td>"
        f"<td>{escape(str(inputs.get('candidate_id', '')))}</td>"
        f"<td>{escape(str(inputs.get('model_id', '')))}</td>"
        f'<td class="{escape(outcome)}">{escape(outcome)}</td>'
        f"<td><details><summary>Inspect action assessment</summary>{assessment}"
        f'<details><summary>Raw action trace</summary><pre tabindex="0"><code>{escape(detail)}</code></pre></details>'
        "</details></td>"
        "</tr>"
    )


def _action_assessment(raw_result: object) -> str:
    result = raw_result if isinstance(raw_result, dict) else {}
    raw_comparisons = result.get("comparisons")
    comparisons = (
        [comparison for comparison in raw_comparisons if isinstance(comparison, dict)]
        if isinstance(raw_comparisons, list)
        else []
    )
    rows = "".join(_comparison_row(comparison) for comparison in comparisons)
    raw_unexpected = result.get("unexpected_actions")
    unexpected = raw_unexpected if isinstance(raw_unexpected, list) else []
    unexpected_items = "".join(f"<li>{_action_text(action)}</li>" for action in unexpected)
    unexpected_panel = f"<h4>Unexpected actions</h4><ul>{unexpected_items}</ul>" if unexpected_items else ""
    return (
        f'<section class="assessment"><h3>Action assessment: {escape(str(result.get("reason", "unknown")))}</h3>'
        '<div class="table-wrap"><table><thead><tr>'
        '<th scope="col">Expected service / target / data</th>'
        '<th scope="col">Actual service / target / data</th>'
        '<th scope="col">Service match</th><th scope="col">Target match</th><th scope="col">Data match</th>'
        f"</tr></thead><tbody>{rows}</tbody></table></div>{unexpected_panel}</section>"
    )


def _comparison_row(comparison: dict[str, object]) -> str:
    return (
        "<tr>"
        f"<td>{_action_text(comparison.get('expected'))}</td>"
        f"<td>{_action_text(comparison.get('actual'))}</td>"
        f"<td>{_match_indicator(comparison.get('service_matches'))}</td>"
        f"<td>{_match_indicator(comparison.get('target_matches'))}</td>"
        f"<td>{_match_indicator(comparison.get('service_data_matches'))}</td>"
        "</tr>"
    )


def _action_text(raw_action: object) -> str:
    if not isinstance(raw_action, dict):
        return "<em>none</em>"
    service = f"{raw_action.get('domain', '')}.{raw_action.get('service', '')}".strip(".")
    targets = json.dumps(raw_action.get("target_entity_ids", []), sort_keys=True)
    data = json.dumps(raw_action.get("service_data"), sort_keys=True)
    return f"<code>{escape(service)}</code><br>targets: {escape(targets)}<br>data: {escape(data)}"


def _match_indicator(value: object) -> str:
    matched = value is True
    css_class = "match" if matched else "mismatch"
    label = "yes" if matched else "no"
    return f'<span class="{css_class}">{label}</span>'


def _outcome(case: dict[str, object]) -> str:
    output = case.get("output")
    outcome = output.get("outcome") if isinstance(output, dict) else None
    state = outcome.get("state") if isinstance(outcome, dict) else None
    return str(state or "incomplete")
