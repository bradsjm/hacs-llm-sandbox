"""Self-contained HTML rendering from the immutable saved-report presentation model."""

from dataclasses import asdict
from datetime import UTC, datetime
import html
import json
from pathlib import Path
import shlex
from tempfile import NamedTemporaryFile

from llm_sandbox_evals.presentation import ReportPresentationModel, effective_cause, result_label
from llm_sandbox_evals.reports import MatrixReport, load_report
from llm_sandbox_evals.statistics import canonical_cells, pair_aggregates, wilson_interval


def write_html(run_dir: Path) -> Path:
    """Render a loaded report atomically, with distinct load and render failure pages."""
    report_html = run_dir / "report.html"
    try:
        report = load_report(run_dir)
    except Exception as err:  # noqa: BLE001 - a missing or invalid report cannot be regenerated locally.
        rendered = _invalid_report_html(run_dir, err)
    else:
        try:
            rendered = render_html(report, run_id=run_dir.name)
        except Exception as err:  # noqa: BLE001 - the validated JSON report remains the recovery source of truth.
            # Branch boundary: only a successfully loaded report supports a no-model re-render recovery command.
            rendered = _recovery_html(run_dir, err)
    # State mutation point: replace a prior browser view only after a complete error or report page is available.
    _atomic_text_write(report_html, rendered)
    return report_html


def render_html(report: MatrixReport, *, run_id: str | None = None) -> str:
    """Return a complete HTML document projected from an immutable report model."""
    model = ReportPresentationModel.from_report(report)
    created_at = _created_at(run_id)
    descriptor = model.descriptor
    counts = model.canonical_counts
    quality_interval = model.canonical_quality_interval
    aggregates = pair_aggregates(canonical_cells(model.cells))
    cells = [
        {
            "case_id": cell.case_id,
            "category": cell.category,
            "candidate_id": cell.candidate_id,
            "model_id": cell.model_id,
            "variant": cell.variant,
            "result": result_label(cell.trace),
            "cause": effective_cause(cell.trace),
            "state": cell.trace.outcome.state,
            "scoring_mode": cell.trace.outcome.scoring_mode,
            "score_reason": cell.trace.outcome.score_reason,
            "desired_entities": cell.trace.desired_entities,
            "end_state_result": cell.trace.end_state_result,
            "request": cell.trace.request_text,
            "diagnostics": asdict(cell.trace.diagnostics),
            "metrics": cell.metrics,
            "action_result": cell.trace.action_result,
            "action_ledger": cell.trace.action_ledger,
            "tool_events": cell.trace.tool_events,
            "answer": cell.trace.answer,
        }
        for cell in model.cells
    ]
    payload = json.dumps(
        {
            "run_id": run_id,
            "created_at": created_at,
            "descriptor": descriptor,
            "counts": {
                "total": counts.total,
                "scored": counts.scored,
                "correct": counts.correct,
                "incorrect": counts.incorrect,
                "incomplete": counts.incomplete,
                "quality_rate": counts.quality_rate,
                "coverage_rate": counts.coverage_rate,
                "quality_interval": quality_interval,
            },
            "issues": dict(model.operational_issues),
            "aggregates": [
                {
                    "candidate": aggregate.candidate_id,
                    "variant": aggregate.variant,
                    "quality_rate": aggregate.counts.quality_rate,
                    "quality_interval": wilson_interval(aggregate.counts.correct, aggregate.counts.scored),
                    "coverage_rate": aggregate.counts.coverage_rate,
                    "scored": aggregate.counts.scored,
                    "total": aggregate.counts.total,
                    "calls": aggregate.mean_calls,
                    "failures": aggregate.mean_failed_calls,
                    "elapsed": aggregate.mean_elapsed,
                    "tokens": aggregate.total_tokens,
                    "cost": aggregate.total_cost,
                }
                for aggregate in aggregates
            ],
            "category_aggregates": [
                {
                    "candidate": aggregate.candidate_id,
                    "variant": aggregate.variant,
                    "category": aggregate.category,
                    "quality_rate": aggregate.counts.quality_rate,
                    "coverage_rate": aggregate.counts.coverage_rate,
                    "scored": aggregate.counts.scored,
                }
                for aggregate in model.category_aggregates
            ],
            "cells": cells,
        },
        default=_json_default,
        ensure_ascii=False,
    ).replace("</", "<\\/")
    return _PAGE.replace("__REPORT_DATA__", payload)


def _created_at(run_id: str | None) -> str:
    """Derive a human timestamp from the stable run id when possible."""
    if run_id is None:
        return "—"
    try:
        return datetime.strptime(run_id, "%Y%m%d-%H%M%S-%f").replace(tzinfo=UTC).isoformat()
    except ValueError:
        return "—"


def _json_default(value: object) -> object:
    """Serialize frozen trace dataclasses embedded in the immutable page model."""
    if hasattr(value, "__dataclass_fields__"):
        return {name: getattr(value, name) for name in value.__dataclass_fields__}
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _atomic_text_write(path: Path, content: str) -> None:
    """Atomically replace HTML so readers see either a complete prior or new page."""
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            # State mutation point: retain the tempfile path before content writing can fail.
            temporary = Path(handle.name)
            handle.write(content)
        temporary.replace(path)
    finally:
        # Branch boundary: clean up a partial temp page on every write, close, or replace failure.
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _recovery_html(run_dir: Path, error: Exception) -> str:
    """Return a small durable recovery page without pretending a partial is a report."""
    command = (
        "python -m llm_sandbox_evals report "
        f"{shlex.quote(run_dir.name)} --runs-dir {shlex.quote(str(run_dir.parent))} --html"
    )
    return f"""<!doctype html><meta charset=\"utf-8\"><title>Eval report recovery</title>
<main><h1>Report render failed</h1><p>{html.escape(str(error))}</p>
<p>The native <code>report.json</code> is still valid at <code>{html.escape(str(run_dir))}</code>.</p>
<pre>{html.escape(command)}</pre></main>"""


def _invalid_report_html(run_dir: Path, error: Exception) -> str:
    """Return a load/validation error page without claiming a re-render can repair the input."""
    return f"""<!doctype html><meta charset=\"utf-8\"><title>Eval report unavailable</title>
<main><h1>Report could not be loaded</h1><p>{html.escape(str(error))}</p>
<p>Inspect <code>{html.escape(str(run_dir / "report.json"))}</code> or rerun the evaluation.</p></main>"""


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Sandbox Eval Report</title><style>
body{font:15px system-ui,sans-serif;margin:0;background:#f4f7fb;color:#142334}main{max-width:1400px;margin:auto;padding:1rem}section,article{background:white;border:1px solid #d4dfeb;border-radius:8px;padding:1rem;margin:1rem 0}.hero{background:#075985;color:#fff}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.75rem}.card{background:#fff2;border:1px solid #fff5;padding:.7rem;border-radius:6px}.card strong{display:block;font-size:1.45rem}table{width:100%;border-collapse:collapse}th,td{padding:.45rem;border-bottom:1px solid #d4dfeb;text-align:left;vertical-align:top}button{cursor:pointer}pre{white-space:pre-wrap;overflow:auto}.incomplete{color:#8a6100}.incorrect{color:#b42318}.correct{color:#17803a}.detail{display:none}.detail:target{display:block}.muted{color:#526779}</style></head>
<body><main><header class="hero"><p>LLM Sandbox eval matrix</p><h1 id="run-title">Eval report</h1><p id="variant-config"></p><div class="grid"><div class="card"><strong id="quality">—</strong>Canonical quality · <span id="quality-ci">—</span> Wilson 95% CI</div><div class="card"><strong id="coverage">—</strong>Canonical coverage</div><div class="card"><strong id="incomplete">—</strong>Incomplete</div><div class="card"><strong id="total">—</strong>Total cells</div><div class="card"><strong id="candidate-variants">—</strong>Candidate x variants</div></div></header>
<section><h2>Canonical candidate comparison</h2><table id="comparison"><thead><tr><th>Candidate</th><th>Variant</th><th>Quality</th><th>Wilson 95% CI</th><th>Coverage</th><th>Calls/failures</th><th>Avg elapsed</th><th>Tokens/cost</th></tr></thead><tbody></tbody></table></section>
<section><h2>By category</h2><table id="categories"><thead><tr><th>Candidate</th><th>Variant</th><th>Category</th><th>Quality</th><th>Coverage</th><th>Scored</th></tr></thead><tbody></tbody></table></section>
<section><h2>Charts</h2><article><h3>Candidate x variant correct-rate heatmap</h3><table id="heatmap"></table></article><div class="grid"><article><h3>Quality (scored cells)</h3><table id="quality-chart"></table></article><article><h3>Operational failures</h3><table id="failure-chart"></table></article></div></section>
<section><h2>Cells</h2><button id="export-csv" type="button">Export CSV</button><table id="cells"><thead><tr><th>Case</th><th>Category</th><th>Candidate</th><th>Variant</th><th>Result</th><th>Tools</th><th>Elapsed</th></tr></thead><tbody></tbody></table></section><section id="inspector"><h2>Inspector</h2><p class="muted">Choose a cell to inspect verdict, operational context, end-state evidence, action evidence, tool evidence, answer, and raw details.</p></section></main>
<script type="application/json" id="report-data">__REPORT_DATA__</script><script>
const d=JSON.parse(document.getElementById('report-data').textContent), $=id=>document.getElementById(id), esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const percent=v=>v==null?'—':`${(v*100).toFixed(1)}%`, interval=v=>!v||v[0]==null||v[1]==null?'—':`[${percent(v[0])}, ${percent(v[1])}]`, metric=(c,n)=>c.metrics?.[n]??c.diagnostics?.usage?.[n]??null;
$('run-title').textContent=`Run ${d.run_id??'—'}`;$('quality').textContent=percent(d.counts.quality_rate);$('quality-ci').textContent=interval(d.counts.quality_interval);$('coverage').textContent=percent(d.counts.coverage_rate);$('incomplete').textContent=d.counts.incomplete;$('total').textContent=d.counts.total;
$('candidate-variants').textContent=d.aggregates.length;
$('variant-config').textContent=(d.descriptor.models||[]).map(m=>`${m.variant_label} · temperature=${m.temperature??'default'}`).join(' | ')||'Variant configuration unavailable';
$('comparison').querySelector('tbody').innerHTML=d.aggregates.map(a=>`<tr><td>${esc(a.candidate)}</td><td>${esc(a.variant)}</td><td>${percent(a.quality_rate)}</td><td>${interval(a.quality_interval)}</td><td>${percent(a.coverage_rate)}</td><td>${a.calls.toFixed(1)}/${a.failures.toFixed(1)}</td><td>${a.elapsed.toFixed(2)}s</td><td>${a.tokens??'unavailable'}/${a.cost??'unavailable'}</td></tr>`).join('');
$('categories').querySelector('tbody').innerHTML=d.category_aggregates.map(a=>`<tr><td>${esc(a.candidate)}</td><td>${esc(a.variant)}</td><td>${esc(a.category)}</td><td>${percent(a.quality_rate)}</td><td>${percent(a.coverage_rate)}</td><td>${a.scored}</td></tr>`).join('');
$('heatmap').innerHTML=`<tr><th>Candidate / variant</th><th>Correct rate</th></tr>`+d.aggregates.map(a=>`<tr><th>${esc(a.candidate)} / ${esc(a.variant)}</th><td${a.quality_rate==null?'':` style="background:rgb(${Math.round(255*(1-a.quality_rate))} ${Math.round(150+105*a.quality_rate)} 120 / .35)"`}>${percent(a.quality_rate)}</td></tr>`).join('');
$('quality-chart').innerHTML=d.aggregates.map(a=>`<tr><th>${esc(a.candidate)} / ${esc(a.variant)}</th><td>${percent(a.quality_rate)} (${a.scored} scored)</td></tr>`).join('');
$('failure-chart').innerHTML=Object.entries(d.issues).map(([cause,count])=>`<tr><th>${esc(cause)}</th><td>${count}</td></tr>`).join('')||'<tr><td>No operational failures</td></tr>';
const show=c=>{const id='detail';let x=$(id);if(!x){x=document.createElement('article');x.id=id;$('inspector').append(x)}const usage={...c.diagnostics.usage,...c.metrics};x.innerHTML=`<h3>${esc(c.case_id)} <span class="${esc(c.state)}">${esc(c.result)}</span></h3><h4>Operational context</h4><pre>${esc(JSON.stringify({cause:c.cause,scoring_mode:c.scoring_mode,score_reason:c.score_reason,cap_exhausted:c.diagnostics.cap_exhausted,elapsed_seconds:c.diagnostics.elapsed_seconds,usage},null,2))}</pre><h4>End-state evidence</h4><pre>${esc(JSON.stringify({desired_entities:c.desired_entities,end_state_result:c.end_state_result},null,2))}</pre><h4>Action evidence</h4><pre>${esc(JSON.stringify({assessment:c.action_result,ledger:c.action_ledger},null,2))}</pre><h4>Tool evidence</h4><pre>${esc(JSON.stringify(c.tool_events,null,2))}</pre><h4>Answer</h4><pre>${esc(c.answer)}</pre><details><summary>Raw details</summary><pre>${esc(JSON.stringify(c,null,2))}</pre></details>`;x.scrollIntoView({block:'nearest'})};
$('cells').querySelector('tbody').innerHTML=d.cells.map((c,i)=>`<tr><td><button data-cell="${i}">${esc(c.case_id)}</button></td><td>${esc(c.category)}</td><td>${esc(c.candidate_id)}</td><td>${esc(c.variant)}</td><td class="${esc(c.state)}">${esc(c.result)}</td><td>${c.diagnostics.tool_calls}</td><td>${c.diagnostics.elapsed_seconds??'—'}</td></tr>`).join('');document.addEventListener('click',e=>{const i=e.target.dataset.cell;if(i!==undefined)show(d.cells[Number(i)])});
$('export-csv').addEventListener('click',()=>{const rows=[['case_id','category','candidate_id','model_id','variant','outcome','cause','score','tool_calls','elapsed_seconds','total_tokens','cost'],...d.cells.map(c=>[c.case_id,c.category,c.candidate_id,c.model_id,c.variant,c.state,c.cause,c.state==='correct'?1:0,c.diagnostics.tool_calls,c.diagnostics.elapsed_seconds??'',metric(c,'total_tokens')??'',metric(c,'cost')??''])];const csv=rows.map(r=>r.map(v=>`"${String(v).replaceAll('"','""')}"`).join(',')).join('\\n');const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));link.download=`eval-${d.run_id??'report'}-cells.csv`;link.click();URL.revokeObjectURL(link.href)});
</script></body></html>"""
