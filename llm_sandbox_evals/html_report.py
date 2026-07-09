"""Interactive HTML rendering for native eval report artifacts."""

import json
from datetime import UTC, datetime
from pathlib import Path


def write_html(run_dir: Path) -> Path:
    """Render report.json in run_dir into report.html and return its path.

    Derives run_id from the directory name and created_at by parsing run_id
    as a '%Y%m%d-%H%M%S-%f' UTC timestamp; falls back to report.json mtime.
    """
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
    """Return a complete self-contained HTML document string for one native report dict."""
    payload = {**report, "_meta": {"run_id": run_id, "created_at": created_at}}
    report_data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    return _HTML_TEMPLATE.replace("__REPORT_DATA__", report_data)


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LLM Sandbox Eval Report</title>
  <link rel="icon" href="data:,">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2.0.6/css/pico.min.css" integrity="sha384-7P0NVe9LPDbUCAF+fH2R8Egwz1uqNH83Ns/bfJY0fN2XCDBMUI2S9gGzIOIRBKsA" crossorigin="anonymous">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/ag-grid-community@32.3.3/styles/ag-grid.css" integrity="sha384-gIsz6JTVbF5eW0CcuAljKL2kOE400G4TjBH3CBtGXjf8BkjmC5x9jp4Y+bYa7HKA" crossorigin="anonymous">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/ag-grid-community@32.3.3/styles/ag-theme-alpine.css" integrity="sha384-3blXdgSdMqVhgou73DTX5qLULk1Hwjsux2R8xxmKrAhG2NvA/VNZSnCqy4+FrvEG" crossorigin="anonymous">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/highlight.js@11.10.0/styles/github.min.css" integrity="sha384-eFTL69TLRZTkNfYZOLM+G04821K1qZao/4QLJbet1pP4tcF+fdXq/9CdqAbWRl/L" crossorigin="anonymous">
  <style>
    :root {
      /* Pin Pico's scale: Pico inflates --pico-font-size up to ~131% on wide
         viewports, which is the oversized "blind person" look. Fix the base. */
      --pico-font-size: 100%;
      --pico-line-height: 1.45;
      --pico-spacing: .75rem;
      --pico-block-spacing-vertical: .9rem;
      --pico-typography-spacing-vertical: .75rem;
      --pico-border-radius: .5rem;
      --pico-form-element-spacing-vertical: .5rem;
      --pico-form-element-spacing-horizontal: .75rem;
      --text: #102033;
      --heading: #0f172a;
      --muted: #4b5f73;
      --body-bg: #f4f7fb;
      --panel: #fff;
      --panel-soft: #f8fafc;
      --panel-border: #cbd8e3;
      --focus: #0b7285;
      --hero-start: #102a43;
      --hero-end: #075985;
      --pass: #17803a;
      --fail: #b42318;
      --incomplete: #8a6100;
      --muted-border: var(--panel-border);
      --radius: 10px;
      --card-pad: .85rem 1rem;
      --gap: .9rem;
      --shadow: 0 1px 2px rgb(2 8 23 / 6%), 0 1px 3px rgb(2 8 23 / 8%);
      --shadow-hero: 0 6px 20px rgb(16 42 67 / 14%);
      /* Comfortable ~15px base; overrides Pico's viewport-scaled variable. */
      font-size: 15px;
    }
    body { background: var(--body-bg); color: var(--text); }
    main.container-fluid { width: min(1600px, calc(100% - 2rem)); margin-inline: auto; padding-block: .75rem 3rem; }
    section { margin-block-start: 1rem; }
    h2 { color: var(--heading); font-size: 1.15rem; font-weight: 600; margin-block: 0 .6rem; }
    .panel h3, .analysis-card h3, #detail-panel h3, #detail-panel h4, .tool-card h5 { color: var(--heading); margin-block-start: 0; font-size: .95rem; font-weight: 600; }
    .hero { background: linear-gradient(135deg, var(--hero-start), var(--hero-end)); color: #fff; border-radius: var(--radius); padding: clamp(.9rem, 1.6vw, 1.25rem); margin-block: .75rem 1rem; box-shadow: var(--shadow-hero); }
    .hero h1, .hero p { color: inherit; }
    .hero h1 { margin-block: .1rem .2rem; font-size: clamp(1.3rem, 2vw, 1.6rem); }
    .hero p { margin-block: 0; opacity: .9; font-size: .9rem; }
    .summary-grid, .chart-row, .filter-grid, .detail-grid { display: grid; gap: var(--gap); }
    .summary-grid { grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); margin-block-start: .8rem; }
    .summary-card, .panel, .analysis-card, .action-card, .tool-card { background: var(--panel); border: 1px solid var(--panel-border); border-radius: var(--radius); padding: var(--card-pad); box-shadow: var(--shadow); color: var(--text); min-inline-size: 0; }
    .hero .summary-card { background: #fff; border-color: #d8e2eb; }
    .summary-card strong { display: block; color: var(--heading); font-size: 1.55rem; line-height: 1.1; letter-spacing: -.02em; }
    .summary-card span { color: var(--muted); font-size: .8rem; font-weight: 600; }
    .chart-row { grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr)); align-items: stretch; margin-block-start: var(--gap); }
    .chart { min-block-size: 280px; inline-size: 100%; }
    #heatmap { min-block-size: clamp(200px, 22vw, 300px); }
    .analysis-grid { display: flex; flex-direction: column; gap: var(--gap); }
    .analysis-card p, #detail-panel p, .gate p { color: var(--muted); }
    .kpi-row { display: flex; flex-wrap: wrap; gap: var(--gap); }
    .kpi-chip { background: var(--panel-soft); border: 1px solid var(--panel-border); border-radius: var(--radius); padding: .5rem .8rem; min-inline-size: 0; }
    .kpi-chip span { display: block; color: var(--muted); font-size: .78rem; font-weight: 600; }
    .kpi-chip strong { color: var(--heading); font-size: 1.1rem; }
    .table-wrap { overflow-x: auto; scrollbar-width: thin; scrollbar-color: #b7c6d6 transparent; scrollbar-gutter: stable; overscroll-behavior: contain; border: 1px solid var(--panel-border); border-radius: .5rem; }
    .table-wrap table { margin: 0; width: 100%; font-size: .85rem; }
    .table-wrap th { background: var(--panel-soft); color: var(--heading); font-weight: 700; border-block-end: 1px solid var(--panel-border); }
    .table-wrap th, .table-wrap td { padding: .45rem .65rem; white-space: nowrap; color: var(--text); }
    .table-wrap tbody tr:nth-child(even) { background: var(--panel-soft); }
    .filter-grid { grid-template-columns: repeat(auto-fit, minmax(min(100%, 160px), 1fr)); align-items: end; }
    .filter-grid label { margin: 0; color: var(--heading); font-weight: 600; font-size: .85rem; min-inline-size: 0; }
    #filters :is(select, input[type="search"]) { background-color: #fff; color: var(--text); border: 1px solid #b7c6d6; border-radius: .5rem; box-shadow: none; min-block-size: 2.3rem; font-size: .85rem; }
    #filters :is(select, input[type="search"]):focus-visible, #filters input[type="checkbox"]:focus-visible, pre:focus-visible, summary:focus-visible { outline: 3px solid rgb(11 114 133 / 35%); outline-offset: 2px; border-color: var(--focus); }
    #filters input[type="search"]::placeholder { color: #64748b; opacity: 1; }
    #filters input[role="switch"] { accent-color: var(--focus); }
    #results-grid { block-size: clamp(480px, 60vh, 720px); min-block-size: 480px; inline-size: 100%; border: 1px solid var(--panel-border); border-radius: var(--radius); overflow: hidden; }
    .ag-theme-alpine {
      --ag-font-size: 13px;
      --ag-font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --ag-foreground-color: var(--text);
      --ag-secondary-foreground-color: var(--muted);
      --ag-background-color: #fff;
      --ag-header-background-color: #f1f5f9;
      --ag-header-foreground-color: var(--heading);
      --ag-border-color: var(--panel-border);
      --ag-row-hover-color: #eaf6f8;
      --ag-selected-row-background-color: #dff4f7;
      --ag-range-selection-border-color: var(--focus);
      --ag-wrapper-border-radius: 10px;
      --ag-row-height: 36px;
      --ag-header-height: 40px;
      scrollbar-gutter: stable;
      overscroll-behavior: contain;
    }
    .ag-theme-alpine .ag-header-cell-label { font-weight: 700; }
    .ag-theme-alpine .ag-full-width-row { background: #f8fafc; border-block: 1px solid var(--panel-border); padding-inline: .75rem; }
    .badge { display: inline-flex; align-items: center; border-radius: 999px; padding: .12rem .42rem; font-weight: 700; font-size: .7rem; line-height: 1; color: #fff; text-transform: uppercase; letter-spacing: .03em; vertical-align: middle; }
    .badge.pass { background: var(--pass); } .badge.fail { background: var(--fail); } .badge.incomplete { background: var(--incomplete); }
    .gate { background: var(--panel); border: 1px solid var(--panel-border); border-radius: var(--radius); padding: var(--card-pad); box-shadow: var(--shadow); border-inline-start: .35rem solid var(--muted-border); margin-block: .5rem; }
    .gate.pass { border-inline-start-color: var(--pass); } .gate.fail { border-inline-start-color: var(--fail); }
    .gate-title, .card-title { display: flex; justify-content: space-between; gap: 1rem; align-items: baseline; }
    .required { color: #8a6100; font-size: .82rem; }
    .empty-state { color: var(--muted); padding: 1rem; border: 1px dashed var(--panel-border); border-radius: var(--radius); background: #fff; }
    pre { max-block-size: 32rem; overflow: auto; scrollbar-width: thin; scrollbar-color: #b7c6d6 transparent; scrollbar-gutter: stable; overscroll-behavior: contain; }
    code { white-space: pre; }
    details { margin-block: 1rem; }
    .error-card { border-color: var(--fail); background: #fff5f5; color: #7a271a; }
    .ok { border-inline-start: .35rem solid var(--pass); } .error { border-inline-start: .35rem solid var(--fail); }
    @media (max-width: 760px) { main.container-fluid { width: min(100% - 1rem, 1600px); } #results-grid { min-block-size: 460px; } }
  </style>
</head>
<body>
  <main class="container-fluid">
    <header class="hero" id="header-band">
      <p>LLM Sandbox eval matrix</p>
      <h1 id="run-title">Eval report</h1>
      <p id="created-at">Created: —</p>
      <section class="summary-grid" aria-label="Run summary">
        <article class="summary-card"><strong id="overall-mean">—</strong><span>Overall mean score</span></article>
        <article class="summary-card"><strong id="case-count">0</strong><span>Cells</span></article>
        <article class="summary-card"><strong id="incomplete-count">0</strong><span>Incomplete cells</span></article>
        <article class="summary-card"><strong id="candidate-count">0</strong><span>Candidates &times; models</span></article>
      </section>
    </header>

    <section id="charts" aria-labelledby="charts-heading">
      <h2 id="charts-heading">Visual navigation</h2>
      <div id="chart-empty" class="empty-state" hidden>No cases are available in this report.</div>
      <article class="panel"><h3>Candidate &times; model mean score</h3><div id="heatmap" class="chart"></div></article>
      <div class="chart-row">
        <article class="panel"><h3>Outcome by category</h3><div id="outcomes" class="chart"></div></article>
        <article class="panel"><h3>Score distribution</h3><div id="histogram" class="chart"></div></article>
        <article class="panel"><h3>Tool calls by category</h3><div id="toolcalls" class="chart"></div></article>
      </div>
    </section>

    <section id="analyses" aria-labelledby="analyses-heading">
      <h2 id="analyses-heading">Native analyses</h2>
      <div id="analysis-list" class="analysis-grid"></div>
    </section>

    <section id="results" aria-labelledby="results-heading">
      <h2 id="results-heading">Results grid</h2>
      <search class="panel">
        <form id="filters" class="filter-grid">
          <label>Candidate<select id="candidate-filter"><option value="">All</option></select></label>
          <label>Model<select id="model-filter"><option value="">All</option></select></label>
          <label>Category<select id="category-filter"><option value="">All</option></select></label>
          <label>Status<select id="status-filter"><option value="">All</option><option value="pass">Pass</option><option value="fail">Fail</option><option value="incomplete">Incomplete</option></select></label>
          <label>Search<input id="quick-filter" type="search" placeholder="Case name or id"></label>
          <label><input id="group-case" type="checkbox" role="switch"> Group by case</label>
        </form>
      </search>
      <div id="results-grid" class="ag-theme-alpine"></div>
    </section>

    <section id="detail" aria-labelledby="detail-heading">
      <h2 id="detail-heading">Cell detail</h2>
      <div id="detail-panel" class="empty-state">Select a row to inspect checks, tool calls, actions, final answer, and raw JSON.</div>
    </section>
  </main>

  <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js" integrity="sha384-Mx5lkUEQPM1pOJCwFtUICyX45KNojXbkWdYhkKUKsbv391mavbfoAmONbzkgYPzR" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/ag-grid-community@32.3.3/dist/ag-grid-community.min.js" integrity="sha384-OEiOpwgvTUxOUC72SwfU8zDqZrWBwr0YNPNhovPbjCxdaTD0cLxMHBew/o8zuRln" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.10.0/build/highlight.min.js" integrity="sha384-GdEWAbCjn+ghjX0gLx7/N1hyTVmPAjdC2OvoAA0RyNcAOhqwtT8qnbCxWle2+uJX" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.10.0/build/languages/python.min.js" integrity="sha384-YDj7s2Wf0QEwarV3OB8lvoNJJCH032vOLMDo2HDrYiEpQ+QmKN+e++x3hElX5S+w" crossorigin="anonymous"></script>
  <script type="application/json" id="report-data">__REPORT_DATA__</script>
  <script>
    (() => {
      const fmtNumber = value => typeof value === "number" ? value.toFixed(3) : value;
      const escapeForCode = value => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
      const codeBlock = (value, lang) => `<pre tabindex="0"><code class="language-${lang}">${escapeForCode(value)}</code></pre>`;

      const app = () => {
        const DATA = JSON.parse(document.getElementById("report-data")?.textContent || "{}");
        const CELLS = (DATA.cases || []).map(c => {
          const t = c.output || {};
          const checks = Array.isArray(t.checks) ? t.checks : [];
          const reqFail = checks.find(ch => ch.required && !ch.passed);
          const incomplete = checks.some(ch => ch.name === "model_error");
          return {
            case_id: c.inputs?.case_id ?? t.case_id,
            category: c.inputs?.category ?? t.category,
            candidate_id: c.inputs?.candidate_id ?? t.candidate_id,
            model_id: c.inputs?.model_id ?? t.model_id,
            score: (typeof t.score === "number") ? t.score : (c.scores?.score?.value ?? 0),
            tool_calls: t.tool_call_count ?? 0,
            checks, trace: t, name: c.name, raw: c,
            _status: incomplete ? "incomplete" : (reqFail ? "fail" : "pass"),
            _firstFail: reqFail ? reqFail.name : "—",
          };
        });
        const unique = values => [...new Set(values.filter(v => v !== undefined && v !== null && v !== ""))].sort();
        const candidates = unique(CELLS.map(c => c.candidate_id));
        const models = unique(CELLS.map(c => c.model_id));
        const categories = unique(CELLS.map(c => c.category));
        const byTitle = title => (Array.isArray(DATA.analyses) ? DATA.analyses : []).find(a => a?.title === title);
        const fmt = value => typeof value === "number" ? value.toFixed(3) : (value ?? "—");
        const text = value => String(value ?? "—");
        const escapeHtml = value => text(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
        const json = value => JSON.stringify(value ?? null, null, 2);

        document.title = `LLM Sandbox Eval Report ${DATA._meta?.run_id ?? ""}`.trim();
        document.getElementById("run-title").textContent = DATA._meta?.run_id ? `Run ${DATA._meta.run_id}` : "Eval report";
        const created = DATA._meta?.created_at ? new Date(DATA._meta.created_at) : null;
        document.getElementById("created-at").textContent = `Created: ${created && !Number.isNaN(created.valueOf()) ? created.toLocaleString() : "—"}`;
        document.getElementById("overall-mean").textContent = fmt(byTitle("Overall mean score")?.value);
        document.getElementById("case-count").textContent = String(CELLS.length);
        document.getElementById("incomplete-count").textContent = String(byTitle("Incomplete cells")?.value ?? CELLS.filter(c => c._status === "incomplete").length);
        document.getElementById("candidate-count").textContent = `${candidates.length} x ${models.length}`;

        renderAnalyses(DATA.analyses || [], escapeHtml);
        renderCharts(CELLS, candidates, models, categories);
        renderGrid(CELLS, candidates, models, categories, escapeHtml, json);
      };

      const renderAnalyses = (analyses, escapeHtml) => {
        const host = document.getElementById("analysis-list");
        const list = Array.isArray(analyses) ? analyses : [];
        // "Overall mean score" and "Incomplete cells" are already surfaced in the hero
        // summary band; skip them here so they are not duplicated as scalar cards.
        const heroScalars = new Set(["Overall mean score", "Incomplete cells"]);
        // Only render a description when it carries real text (avoids a stray "—").
        const desc = value => (typeof value === "string" && value.trim()) ? `<p>${escapeHtml(value)}</p>` : "";
        const scalars = list.filter(a => a?.type === "scalar" && !heroScalars.has(a.title));
        const tables = list.filter(a => a?.type === "table" && Array.isArray(a.columns) && Array.isArray(a.rows));
        // Branch boundary: nothing left to show once hero-duplicated scalars are removed.
        if (scalars.length === 0 && tables.length === 0) {
          host.innerHTML = '<div class="empty-state">No additional native analyses to show.</div>';
          return;
        }
        // Any remaining (non-hero) scalar analyses render as a compact KPI strip.
        const chips = scalars.length
          ? `<div class="kpi-row">${scalars.map(a => `<article class="kpi-chip"><span>${escapeHtml(a.title)}</span><strong>${escapeHtml(a.value)} ${escapeHtml(a.unit || "")}</strong></article>`).join("")}</div>`
          : "";
        const tableCards = tables.map(analysis => {
          const head = analysis.columns.map(col => `<th>${escapeHtml(col)}</th>`).join("");
          const rows = analysis.rows.map(row => `<tr>${(Array.isArray(row) ? row : [row]).map(cell => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`).join("");
          return `<article class="analysis-card table-analysis"><h3>${escapeHtml(analysis.title)}</h3>${desc(analysis.description)}<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div></article>`;
        }).join("");
        host.innerHTML = chips + tableCards;
      };

      const renderCharts = (cells, candidates, models, categories) => {
        if (cells.length === 0 || !window.echarts) {
          document.getElementById("chart-empty").hidden = false;
          return;
        }
        const chartEls = ["heatmap", "outcomes", "histogram", "toolcalls"].map(id => document.getElementById(id));
        const charts = chartEls.map(el => echarts.init(el));
        const mean = values => values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
        // Five-number summary [min, Q1, median, Q3, max] via linear-interpolation quantiles.
        const quantile = (sorted, p) => {
          if (sorted.length === 0) return 0;
          const pos = (sorted.length - 1) * p;
          const base = Math.floor(pos);
          const rest = pos - base;
          return sorted[base + 1] !== undefined ? sorted[base] + rest * (sorted[base + 1] - sorted[base]) : sorted[base];
        };
        const boxStats = values => {
          if (values.length === 0) return null;
          const sorted = [...values].sort((a, b) => a - b);
          return [sorted[0], quantile(sorted, 0.25), quantile(sorted, 0.5), quantile(sorted, 0.75), sorted[sorted.length - 1]];
        };
        // Mirror experiment._complete_cases / CandidateMatrixReport: incomplete
        // (model_error) cells are excluded from mean-score denominators so a provider
        // outage does not read as a near-zero candidate/model quality.
        const completeCells = cells.filter(c => c._status !== "incomplete");
        const heatData = [];
        candidates.forEach((candidate, y) => models.forEach((model, x) => {
          heatData.push([x, y, Number(mean(completeCells.filter(c => c.candidate_id === candidate && c.model_id === model).map(c => c.score)).toFixed(3))]);
        }));
        const chartText = { color: "#102033", fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" };
        const axisLabel = { color: "#334155", fontSize: 11, hideOverlap: true };
        const axisLine = { lineStyle: { color: "#cbd8e3" } };
        const splitLine = { lineStyle: { color: "#e2e8f0" } };
        charts[0].setOption({ textStyle: chartText, tooltip: { position: "top" }, grid: { left: 120, right: 32, top: 20, bottom: 76, containLabel: true }, xAxis: { type: "category", data: models, axisLabel: { ...axisLabel, rotate: models.length > 1 ? 25 : 0 }, axisLine }, yAxis: { type: "category", data: candidates, axisLabel, axisLine }, visualMap: { min: 0, max: 1, orient: "horizontal", left: "center", bottom: 0, textStyle: { color: "#334155" }, inRange: { color: ["#b42318", "#f2c94c", "#17803a"] } }, series: [{ type: "heatmap", data: heatData, label: { show: true, color: "#0f172a", fontWeight: 700 } }] });
        const statusNames = ["pass", "fail", "incomplete"];
        charts[1].setOption({ textStyle: chartText, tooltip: { trigger: "axis", axisPointer: { type: "shadow" } }, legend: { textStyle: { color: "#334155" } }, grid: { left: 44, right: 18, top: 36, bottom: 74, containLabel: true }, xAxis: { type: "category", data: categories, axisLabel: { ...axisLabel, rotate: categories.length > 3 ? 25 : 0 }, axisLine }, yAxis: { type: "value", axisLabel, axisLine, splitLine }, series: statusNames.map(status => ({ name: status, type: "bar", stack: "outcome", data: categories.map(cat => cells.filter(c => c.category === cat && c._status === status).length) })) });
        const bins = Array.from({ length: 10 }, (_, i) => ({ label: `${(i / 10).toFixed(1)}-${((i + 1) / 10).toFixed(1)}`, count: 0 }));
        cells.forEach(cell => { bins[Math.min(9, Math.max(0, Math.floor(cell.score * 10)))].count += 1; });
        charts[2].setOption({ textStyle: chartText, tooltip: {}, grid: { left: 44, right: 18, top: 20, bottom: 74, containLabel: true }, xAxis: { type: "category", data: bins.map(b => b.label), axisLabel: { ...axisLabel, rotate: 30 }, axisLine }, yAxis: { type: "value", axisLabel, axisLine, splitLine }, series: [{ type: "bar", data: bins.map(b => b.count), itemStyle: { color: "#0b7285" } }] });
        // Tool-call efficiency is a first-class scoring dimension: show the per-category
        // distribution rather than a flat mean so par-vs-actual spread is visible.
        const boxCategories = categories.filter(cat => cells.some(c => c.category === cat));
        const boxData = boxCategories.map(cat => boxStats(cells.filter(c => c.category === cat).map(c => c.tool_calls)));
        charts[3].setOption({ textStyle: chartText, tooltip: { trigger: "item" }, grid: { left: 48, right: 18, top: 26, bottom: 74, containLabel: true }, xAxis: { type: "category", data: boxCategories, axisLabel: { ...axisLabel, rotate: boxCategories.length > 3 ? 25 : 0 }, axisLine }, yAxis: { type: "value", name: "Tool calls", nameTextStyle: { color: "#334155", fontWeight: 700 }, axisLabel, axisLine, splitLine }, series: [{ type: "boxplot", data: boxData, itemStyle: { color: "#486581" } }] });
        let resizeTimer;
        window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => charts.forEach(chart => chart.resize()), 120); });
      };

      const renderGrid = (cells, candidates, models, categories, escapeHtml, json) => {
        const fill = (id, values) => {
          document.getElementById(id).insertAdjacentHTML("beforeend", values.map(value => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join(""));
        };
        fill("candidate-filter", candidates); fill("model-filter", models); fill("category-filter", categories);
        const groupToggle = document.getElementById("group-case");
        const filterCells = () => {
          const query = (document.getElementById("quick-filter")?.value || "").toLowerCase();
          return cells.filter(cell => {
            const selected = {
              candidate: document.getElementById("candidate-filter")?.value || "",
              model: document.getElementById("model-filter")?.value || "",
              category: document.getElementById("category-filter")?.value || "",
              status: document.getElementById("status-filter")?.value || "",
            };
            const matchesFacets = (!selected.candidate || cell.candidate_id === selected.candidate) && (!selected.model || cell.model_id === selected.model) && (!selected.category || cell.category === selected.category) && (!selected.status || cell._status === selected.status);
            const haystack = `${cell.case_id || ""} ${cell.name || ""}`.toLowerCase();
            return matchesFacets && (!query || haystack.includes(query));
          });
        };
        const rowsForGrid = rows => {
          if (!groupToggle.checked) return rows;
          const grouped = [];
          let lastCase = null;
          [...rows].sort((a, b) => String(a.case_id).localeCompare(String(b.case_id)) || String(a.candidate_id).localeCompare(String(b.candidate_id)) || String(a.model_id).localeCompare(String(b.model_id))).forEach(row => {
            if (row.case_id !== lastCase) {
              grouped.push({ _group: true, case_id: row.case_id, category: row.category });
              lastCase = row.case_id;
            }
            grouped.push(row);
          });
          return grouped;
        };
        const scoreStyle = params => {
          const score = Math.max(0, Math.min(1, Number(params.value ?? 0)));
          return { backgroundColor: `hsl(${Math.round(score * 120)} 70% 88%)`, fontWeight: "700" };
        };
        const gridOptions = {
          rowData: rowsForGrid(filterCells()),
          defaultColDef: { sortable: true, resizable: true, minWidth: 96 },
          rowSelection: { type: "single", enableClickSelection: true },
          pagination: true,
          paginationPageSize: 25,
          paginationPageSizeSelector: [25, 50, 100],
          isFullWidthRow: params => Boolean(params.rowNode.data?._group),
          fullWidthCellRenderer: params => `<strong>Case: ${escapeHtml(params.data.case_id)}</strong> <span>${escapeHtml(params.data.category)}</span>`,
          columnDefs: [
            { field: "case_id", headerName: "Case", flex: 2, minWidth: 220 },
            { field: "category", minWidth: 135, maxWidth: 190 },
            { field: "candidate_id", headerName: "Candidate", minWidth: 135, maxWidth: 220 },
            { field: "model_id", headerName: "Model", minWidth: 135, maxWidth: 240 },
            { field: "score", minWidth: 94, maxWidth: 116, valueFormatter: p => fmtNumber(p.value), cellStyle: scoreStyle },
            { field: "tool_calls", headerName: "Tools", minWidth: 92, maxWidth: 112 },
            { field: "_firstFail", headerName: "First failing gate", flex: 1, minWidth: 170 },
            { field: "_status", headerName: "Status", minWidth: 116, maxWidth: 138, cellRenderer: p => `<span class="badge ${escapeHtml(p.value)}">${escapeHtml(p.value)}</span>` },
          ],
          onRowClicked: event => { if (!event.data?._group) renderDetail(event.data, escapeHtml, json); },
          onRowSelected: event => { if (event.node.isSelected() && !event.data?._group) renderDetail(event.data, escapeHtml, json); },
        };
        const api = agGrid.createGrid(document.getElementById("results-grid"), gridOptions);
        // The facet panel + search drive a manual rowData rebuild, which is the
        // single source of filtering; no AG Grid built-in/external filter is needed.
        document.getElementById("filters").addEventListener("input", () => {
          api.setGridOption("rowData", rowsForGrid(filterCells()));
        });
        document.getElementById("group-case").addEventListener("change", () => {
          api.setGridOption("rowData", rowsForGrid(filterCells()));
        });
        // Suppress form submission on Enter so the filter panel never navigates the
        // page to its own URL (a latent bug, and the source of the file:// "unique
        // origin" warning when the report is opened directly from disk).
        document.getElementById("filters").addEventListener("submit", event => event.preventDefault());
      };

      const renderDetail = (cell, escapeHtml, json) => {
        const trace = cell.trace || {};
        const checks = Array.isArray(trace.checks) ? trace.checks : [];
        const toolEvents = Array.isArray(trace.tool_events) ? trace.tool_events : [];
        const actions = Array.isArray(trace.recorded_actions) ? trace.recorded_actions : [];
        const gates = checks.map(check => `<article class="gate ${check.passed ? "pass" : "fail"}"><div class="gate-title"><strong>${check.passed ? "✓" : "✗"} ${escapeHtml(check.name)}</strong>${check.required ? '<span class="required">required</span>' : ""}</div><p>${escapeHtml(check.feedback || "—")}</p>${check.value !== undefined && check.value !== null ? codeBlock(json(check.value), "json") : ""}</article>`).join("") || '<div class="empty-state">No checks recorded.</div>';
        let categoryDetail = "";
        if (["action_allowed", "action_blocked"].includes(cell.category)) {
          categoryDetail = actions.map(actionCard(escapeHtml, json)).join("") || '<div class="empty-state">No actions recorded.</div>';
        } else if (cell.category === "recorder_read") {
          const recorderTools = new Set(["get_history", "get_statistics", "get_logbook"]);
          categoryDetail = toolEvents.filter(ev => recorderTools.has(ev.tool_name)).map(toolCard(escapeHtml, json, false)).join("") || '<div class="empty-state">No recorder tool events recorded.</div>';
        } else {
          categoryDetail = toolEvents.filter(ev => ev.tool_name === "execute_home_code").map(toolCard(escapeHtml, json, true)).join("") || '<div class="empty-state">No execute_home_code events recorded.</div>';
        }
        const panel = document.getElementById("detail-panel");
        panel.className = "panel";
        const finalAnswer = trace.output || "—";
        let finalAnswerBlock = codeBlock(finalAnswer, "plaintext");
        if (typeof finalAnswer === "string") {
          try {
            finalAnswerBlock = codeBlock(json(JSON.parse(finalAnswer)), "json");
          } catch {
            finalAnswerBlock = codeBlock(finalAnswer, "plaintext");
          }
        } else {
          finalAnswerBlock = codeBlock(json(finalAnswer), "json");
        }
        panel.innerHTML = `<h3>${escapeHtml(cell.case_id)} <span class="badge ${escapeHtml(cell._status)}">${escapeHtml(cell._status)}</span></h3><p>${escapeHtml(cell.candidate_id)} / ${escapeHtml(cell.model_id)} · score ${escapeHtml(fmtNumber(cell.score))} · first failing gate ${escapeHtml(cell._firstFail)}</p><h4>Gate ladder</h4>${gates}<h4>Category trace</h4><div class="detail-grid">${categoryDetail}</div><details><summary>Final answer</summary>${finalAnswerBlock}</details><details><summary>Raw case JSON</summary>${codeBlock(json(cell.raw), "json")}</details>`;
        document.querySelectorAll("#detail-panel pre code").forEach(block => hljs.highlightElement(block));
      };

      const actionCard = (escapeHtml, json) => action => {
        const target = action.target ?? action.service_data ?? {};
        const entityIds = target.entity_id ?? action.service_data?.entity_id ?? "—";
        const err = action.error ? `<p><strong>${escapeHtml(action.error.key)}</strong>: ${escapeHtml(action.error.message)}</p>` : "";
        return `<article class="action-card ${action.status === "ok" ? "ok" : "error"}"><div class="card-title"><strong>${escapeHtml(action.domain)}.${escapeHtml(action.service)}</strong><span>${escapeHtml(action.status)}</span></div><p>Targets: ${escapeHtml(Array.isArray(entityIds) ? entityIds.join(", ") : entityIds)}</p>${err}<details><summary>Action JSON</summary>${codeBlock(json(action), "json")}</details></article>`;
      };

      const toolCard = (escapeHtml, json, python) => (event, index) => {
        const body = python ? codeBlock(event.args?.code || "", "python") : codeBlock(json(event.args || {}), "json");
        return `<article class="tool-card"><h5>${index + 1}. ${escapeHtml(event.tool_name)}</h5>${body}<details><summary>Output JSON</summary>${codeBlock(json(event.output), "json")}</details></article>`;
      };

      try { app(); } catch (error) {
        const msg = String(error?.message || error).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
        document.body.insertAdjacentHTML("afterbegin", `<main class="container"><article class="error-card"><h1>Report render failed</h1><p>${msg}</p></article></main>`);
        console.error(error);
      }
    })();
  </script>
</body>
</html>
"""
