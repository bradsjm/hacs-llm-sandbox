"""Self-contained HTML report generation for eval run artifacts."""

import json
from pathlib import Path

from llm_sandbox_evals.reports import (
    candidate_rows,
    load_results,
    load_run_json,
    matrix_rows,
    score_categories,
    trace_filename,
)
from llm_sandbox_evals.schema import CandidateModelScore


def write_html(run_dir: Path) -> Path:
    """Regenerate ``report.html`` from a run directory's saved artifacts."""
    run_id, created_at, case_count, candidate_ids, model_ids, scores = load_run_json(run_dir / "run.json")
    result_rows = load_results(run_dir / "results.jsonl")
    report_path = run_dir / "report.html"
    report_path.write_text(
        render_html(
            run_id=run_id,
            created_at=created_at,
            case_count=case_count,
            candidate_ids=candidate_ids,
            model_ids=model_ids,
            scores=scores,
            result_rows=result_rows,
        ),
        encoding="utf-8",
    )
    return report_path


def render_html(
    *,
    run_id: str,
    created_at: str,
    case_count: int,
    candidate_ids: list[str],
    model_ids: list[str],
    scores: list[CandidateModelScore],
    result_rows: list[dict[str, object]],
) -> str:
    """Render a standalone HTML document for one saved eval run."""
    payload: dict[str, object] = {
        "run_id": run_id,
        "created_at": created_at,
        "case_count": case_count,
        "candidate_ids": candidate_ids,
        "model_ids": model_ids,
        "categories": score_categories(scores),
        "candidate_rows": _candidate_payload(scores, candidate_ids, model_ids),
        "matrix_rows": _matrix_payload(scores, candidate_ids, model_ids),
        "results": _result_payload(result_rows),
    }
    script_data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    return _HTML_TEMPLATE.replace("__REPORT_DATA__", script_data)


def _candidate_payload(
    scores: list[CandidateModelScore], candidate_ids: list[str], model_ids: list[str]
) -> list[dict[str, object]]:
    """Serialize candidate aggregate rows for the browser renderer."""
    return [
        {
            "candidate_id": row.candidate_id,
            "mean": row.mean,
            "min_model": row.min_model,
            "mean_turns": row.mean_turns,
            "prompt_chars": row.prompt_chars,
            "size_ratio": row.size_ratio,
            "categories": row.category_means,
        }
        for row in candidate_rows(scores, candidate_ids, model_ids)
    ]


def _matrix_payload(
    scores: list[CandidateModelScore], candidate_ids: list[str], model_ids: list[str]
) -> list[dict[str, object]]:
    """Serialize candidate/model matrix rows for the browser renderer."""
    return [
        {
            "candidate_id": row.candidate_id,
            "model_id": row.model_id,
            "mean": row.mean,
            "mean_turns": row.mean_turns,
        }
        for row in matrix_rows(scores, candidate_ids, model_ids)
    ]


def _result_payload(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Serialize compact per-case result rows for the browser renderer."""
    payload: list[dict[str, object]] = []
    for row in rows:
        case_id = _text(row, "case_id")
        candidate_id = _text(row, "candidate_id")
        model_id = _text(row, "model_id")
        checks = _checks(row.get("checks"))
        score = _number(row, "score")
        required_failed = any(_bool(check, "required") and not _bool(check, "passed") for check in checks)
        payload.append(
            {
                "case_id": case_id,
                "category": _text(row, "category"),
                "candidate_id": candidate_id,
                "model_id": model_id,
                "score": score,
                "turns": _integer(row, "turns"),
                "par_turns": _integer(row, "par_turns"),
                "status": "fail" if score == 0.0 or required_failed else "pass",
                "checks": checks,
                "trace_href": f"traces/{trace_filename(case_id, model_id, candidate_id)}",
            }
        )
    return payload


def _checks(value: object) -> list[dict[str, object]]:
    """Return JSON-safe check dictionaries from a result row's ``checks`` field."""
    if not isinstance(value, list):
        return []
    checks: list[dict[str, object]] = []
    for item in value:
        if isinstance(item, dict):
            checks.append(
                {
                    "name": _text(item, "name"),
                    "passed": _bool(item, "passed"),
                    "required": _bool(item, "required"),
                }
            )
    return checks


def _text(row: dict[str, object], key: str) -> str:
    """Return one string field from a decoded JSON object."""
    value = row.get(key)
    return value if isinstance(value, str) else ""


def _bool(row: dict[str, object], key: str) -> bool:
    """Return one boolean field from a decoded JSON object."""
    value = row.get(key)
    return value if isinstance(value, bool) else False


def _number(row: dict[str, object], key: str) -> float:
    """Return one numeric field from a decoded JSON object."""
    value = row.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _integer(row: dict[str, object], key: str) -> int:
    """Return one integer field from a decoded JSON object."""
    value = row.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM Sandbox Eval Report</title>
  <style>
    :root { color-scheme: light dark; --ok: #177245; --bad: #b42318; --muted: #64748b; }
    body { font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem; }
    header, section { margin-bottom: 2rem; }
    .meta { color: var(--muted); display: flex; flex-wrap: wrap; gap: .75rem 1.25rem; }
    table { border-collapse: collapse; width: 100%; margin-top: .75rem; }
    th, td { border-bottom: 1px solid color-mix(in srgb, CanvasText 18%, transparent); padding: .45rem .55rem; text-align: left; vertical-align: top; }
    th { cursor: pointer; user-select: none; background: color-mix(in srgb, CanvasText 7%, transparent); }
    tr:hover { background: color-mix(in srgb, CanvasText 4%, transparent); }
    .num { font-variant-numeric: tabular-nums; text-align: right; }
    .badge { border-radius: 999px; color: white; display: inline-block; font-size: .78rem; font-weight: 700; padding: .1rem .45rem; text-transform: uppercase; }
    .pass { background: var(--ok); }
    .fail { background: var(--bad); }
    .checks { color: var(--muted); font-size: .92rem; }
    .checks ul { margin: .35rem 0 0 1.1rem; padding: 0; }
    .filters { align-items: end; display: flex; flex-wrap: wrap; gap: .75rem; }
    .filters label { display: grid; font-size: .9rem; gap: .2rem; }
    select { min-width: 11rem; padding: .25rem; }
    a { color: LinkText; }
  </style>
</head>
<body>
  <header>
    <h1>LLM Sandbox Eval Report</h1>
    <div class="meta" id="meta"></div>
  </header>

  <section>
    <h2>Candidate leaderboard</h2>
    <div id="candidates"></div>
  </section>

  <section>
    <h2>Candidate x model means</h2>
    <div id="matrix"></div>
  </section>

  <section>
    <h2>Per-case results</h2>
    <div class="filters">
      <label>Candidate <select id="filter-candidate"></select></label>
      <label>Model <select id="filter-model"></select></label>
      <label>Category <select id="filter-category"></select></label>
      <label>Status <select id="filter-status"><option value="">All</option><option value="pass">Pass</option><option value="fail">Fail</option></select></label>
      <span class="meta" id="result-count"></span>
    </div>
    <div id="results"></div>
  </section>

  <script id="report-data" type="application/json">__REPORT_DATA__</script>
  <script>
    const data = JSON.parse(document.getElementById("report-data").textContent);
    const expanded = new Set();
    const sortState = new Map();
    let resultSort = { key: "score", direction: -1 };

    function fmt(value) { return Number(value || 0).toFixed(3); }
    function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
    function textCell(value, className = "") {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      if (className) cell.className = className;
      return cell;
    }
    function compareValues(left, right) {
      if (typeof left === "number" && typeof right === "number") return left - right;
      return String(left).localeCompare(String(right));
    }
    function makeTable(containerId, columns, rows, defaultKey, defaultDirection = -1) {
      const container = document.getElementById(containerId);
      clear(container);
      const table = document.createElement("table");
      const header = table.createTHead().insertRow();
      const state = sortState.get(containerId) || { key: defaultKey, direction: defaultDirection };
      sortState.set(containerId, state);
      for (const column of columns) {
        const th = document.createElement("th");
        th.textContent = column.label + (state.key === column.key ? (state.direction > 0 ? " ↑" : " ↓") : "");
        th.addEventListener("click", () => {
          state.direction = state.key === column.key ? -state.direction : -1;
          state.key = column.key;
          makeTable(containerId, columns, rows, defaultKey, defaultDirection);
        });
        header.appendChild(th);
      }
      const body = table.createTBody();
      const sorted = [...rows].sort((a, b) => state.direction * compareValues(columns.find((c) => c.key === state.key).value(a), columns.find((c) => c.key === state.key).value(b)));
      for (const row of sorted) {
        const tr = body.insertRow();
        for (const column of columns) tr.appendChild(column.cell(row));
      }
      container.appendChild(table);
    }
    function fillSelect(id, values) {
      const select = document.getElementById(id);
      clear(select);
      select.appendChild(new Option("All", ""));
      for (const value of [...new Set(values)].sort()) select.appendChild(new Option(value, value));
      select.addEventListener("change", renderResults);
    }
    function resultKey(row) { return `${row.case_id}\u0000${row.model_id}\u0000${row.candidate_id}`; }
    function resultValue(row, key) { return key === "turns" ? row.turns : key === "score" ? row.score : String(row[key] || ""); }
    function renderResults() {
      const filters = {
        candidate_id: document.getElementById("filter-candidate").value,
        model_id: document.getElementById("filter-model").value,
        category: document.getElementById("filter-category").value,
        status: document.getElementById("filter-status").value,
      };
      const rows = data.results.filter((row) => Object.entries(filters).every(([key, value]) => !value || row[key] === value));
      rows.sort((a, b) => resultSort.direction * compareValues(resultValue(a, resultSort.key), resultValue(b, resultSort.key)));
      document.getElementById("result-count").textContent = `${rows.length} of ${data.results.length} rows`;
      const container = document.getElementById("results");
      clear(container);
      const headers = [["case_id", "Case"], ["category", "Category"], ["candidate_id", "Candidate"], ["model_id", "Model"], ["score", "Score"], ["turns", "Turns"], ["status", "Status"], ["trace", "Trace"]];
      const table = document.createElement("table");
      const header = table.createTHead().insertRow();
      for (const [key, label] of headers) {
        const th = document.createElement("th");
        th.textContent = label + (resultSort.key === key ? (resultSort.direction > 0 ? " ↑" : " ↓") : "");
        th.addEventListener("click", () => { resultSort = { key, direction: resultSort.key === key ? -resultSort.direction : -1 }; renderResults(); });
        header.appendChild(th);
      }
      const body = table.createTBody();
      for (const row of rows) {
        const tr = body.insertRow();
        tr.addEventListener("click", () => { const key = resultKey(row); expanded.has(key) ? expanded.delete(key) : expanded.add(key); renderResults(); });
        tr.append(textCell(row.case_id), textCell(row.category), textCell(row.candidate_id), textCell(row.model_id), textCell(fmt(row.score), "num"), textCell(`${row.turns}/${row.par_turns}`, "num"));
        const status = document.createElement("td");
        const badge = document.createElement("span");
        badge.className = `badge ${row.status}`;
        badge.textContent = row.status;
        status.appendChild(badge);
        tr.appendChild(status);
        const trace = document.createElement("td");
        const link = document.createElement("a");
        link.href = row.trace_href;
        link.textContent = "json";
        link.addEventListener("click", (event) => event.stopPropagation());
        trace.appendChild(link);
        tr.appendChild(trace);
        if (expanded.has(resultKey(row))) {
          const detail = body.insertRow();
          const cell = detail.insertCell();
          cell.colSpan = headers.length;
          cell.className = "checks";
          const list = document.createElement("ul");
          for (const check of row.checks) {
            const item = document.createElement("li");
            item.textContent = `${check.passed ? "✓" : "✗"} ${check.name}${check.required ? " (required)" : ""}`;
            list.appendChild(item);
          }
          cell.append("Checks:", list);
        }
      }
      container.appendChild(table);
    }

    document.getElementById("meta").textContent = `Run ${data.run_id} · Created ${data.created_at} · ${data.case_count} case(s)`;
    fillSelect("filter-candidate", data.candidate_ids);
    fillSelect("filter-model", data.model_ids);
    fillSelect("filter-category", data.categories);
    document.getElementById("filter-status").addEventListener("change", renderResults);
    makeTable("candidates", [
      { key: "candidate_id", label: "Candidate", value: (row) => row.candidate_id, cell: (row) => textCell(row.candidate_id) },
      { key: "mean", label: "Mean", value: (row) => row.mean, cell: (row) => textCell(fmt(row.mean), "num") },
      { key: "min_model", label: "MinModel", value: (row) => row.min_model, cell: (row) => textCell(fmt(row.min_model), "num") },
      { key: "mean_turns", label: "Turns", value: (row) => row.mean_turns, cell: (row) => textCell(fmt(row.mean_turns), "num") },
      { key: "prompt_chars", label: "PromptChars", value: (row) => row.prompt_chars, cell: (row) => textCell(row.prompt_chars, "num") },
      { key: "size_ratio", label: "SizeRatio", value: (row) => row.size_ratio, cell: (row) => textCell(fmt(row.size_ratio), "num") },
      ...data.categories.map((category) => ({ key: `cat:${category}`, label: category, value: (row) => row.categories[category] || 0, cell: (row) => textCell(fmt(row.categories[category] || 0), "num") })),
    ], data.candidate_rows, "mean");
    makeTable("matrix", [
      { key: "candidate_id", label: "Candidate", value: (row) => row.candidate_id, cell: (row) => textCell(row.candidate_id) },
      { key: "model_id", label: "Model", value: (row) => row.model_id, cell: (row) => textCell(row.model_id) },
      { key: "mean", label: "Mean", value: (row) => row.mean, cell: (row) => textCell(fmt(row.mean), "num") },
      { key: "mean_turns", label: "Turns", value: (row) => row.mean_turns, cell: (row) => textCell(fmt(row.mean_turns), "num") },
    ], data.matrix_rows, "mean");
    renderResults();
  </script>
</body>
</html>
"""
