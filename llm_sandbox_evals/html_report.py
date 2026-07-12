"""Interactive HTML rendering for native eval report artifacts."""

from datetime import UTC, datetime
import json
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
  <meta name="color-scheme" content="light dark">
  <title>LLM Sandbox Eval Report</title>
  <link rel="icon" href="data:,">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2.0.6/css/pico.min.css" integrity="sha384-7P0NVe9LPDbUCAF+fH2R8Egwz1uqNH83Ns/bfJY0fN2XCDBMUI2S9gGzIOIRBKsA" crossorigin="anonymous">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/ag-grid-community@32.3.3/styles/ag-grid.css" integrity="sha384-gIsz6JTVbF5eW0CcuAljKL2kOE400G4TjBH3CBtGXjf8BkjmC5x9jp4Y+bYa7HKA" crossorigin="anonymous">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/ag-grid-community@32.3.3/styles/ag-theme-alpine.css" integrity="sha384-3blXdgSdMqVhgou73DTX5qLULk1Hwjsux2R8xxmKrAhG2NvA/VNZSnCqy4+FrvEG" crossorigin="anonymous">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.10.0/build/styles/github.min.css" integrity="sha384-eFTL69TLRZTkNfYZOLM+G04821K1qZao/4QLJbet1pP4tcF+fdXq/9CdqAbWRl/L" crossorigin="anonymous" media="(prefers-color-scheme: light)">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.10.0/build/styles/github-dark.min.css" integrity="sha384-wH75j6z1lH97ZOpMOInqhgKzFkAInZPPSPlZpYKYTOqsaizPvhQZmAtLcPKXpLyH" crossorigin="anonymous" media="(prefers-color-scheme: dark)">
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
      /* Scheme-aware token pairs: same slate/cyan hue family in both schemes,
         elevation expressed as whisper-quiet lightness steps. */
      color-scheme: light dark;
      --text: light-dark(#102033, #d3dce6);
      --heading: light-dark(#0f172a, #eef3f8);
      --muted: light-dark(#4b5f73, #90a4b9);
      --body-bg: light-dark(#f4f7fb, #0c141d);
      --panel: light-dark(#fff, #121c28);
      --panel-soft: light-dark(#f8fafc, #182533);
      --panel-border: light-dark(#cbd8e3, #253750);
      --control-border: light-dark(#b7c6d6, #33465e);
      --focus: light-dark(#0b7285, #38b6ca);
      --hero-start: #102a43;
      --hero-end: #075985;
      /* Scheme-aware status colors for text/rails; fixed dark fills for badges
         so white badge text stays legible in both schemes. */
      --pass: light-dark(#17803a, #55c97c);
      --fail: light-dark(#b42318, #f2705f);
      --incomplete: light-dark(#8a6100, #d9a514);
      --pass-fill: #177a38;
      --fail-fill: #b42318;
      --incomplete-fill: #8a6100;
      /* Brighter status accents readable on the fixed dark hero gradient. */
      --pass-on-hero: #6fe09a;
      --fail-on-hero: #ff9b8a;
      --incomplete-on-hero: #ffd66e;
      /* Chart tokens resolved at render time via getComputedStyle. */
      --chart-text: light-dark(#102033, #d3dce6);
      --chart-axis: light-dark(#334155, #9fb2c6);
      --chart-line: light-dark(#cbd8e3, #2b3d54);
      --chart-split: light-dark(#e2e8f0, #22303f);
      --chart-box: light-dark(#486581, #7f9cb8);
      --muted-border: var(--panel-border);
      --radius: 10px;
      --card-pad: .85rem 1rem;
      --gap: .9rem;
      --shadow-hero: 0 6px 20px rgb(16 42 67 / 14%);
      /* Comfortable ~15px base; overrides Pico's viewport-scaled variable. */
      font-size: 15px;
    }
    body { background: var(--body-bg); color: var(--text); }
    main.container-fluid { width: min(1600px, calc(100% - 2rem)); margin-inline: auto; padding-block: .75rem 3rem; }
    section { margin-block-start: 1rem; scroll-margin-block-start: 3.4rem; }
    h1, h2, h3 { text-wrap: balance; }
    h2 { color: var(--heading); font-size: 1.15rem; font-weight: 600; margin-block: 0 .6rem; }
    .panel h3, .analysis-card h3, #detail-panel h3, #detail-panel h4, .tool-card h5 { color: var(--heading); margin-block-start: 0; font-size: .95rem; font-weight: 600; }
     /* Sticky section navigation: the report is long, keep orientation cheap. */
    .report-nav { position: sticky; top: 0; z-index: 20; display: flex; align-items: center; gap: 1rem; padding: .45rem .9rem; margin-block-start: .5rem; border: 1px solid var(--panel-border); border-radius: var(--radius); background: color-mix(in srgb, var(--panel) 82%, transparent); backdrop-filter: blur(8px); }
    .report-nav strong { color: var(--heading); font-size: .85rem; margin-inline-end: auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .report-nav a { color: var(--muted); font-size: .85rem; font-weight: 600; text-decoration: none; }
    .report-nav a:hover, .report-nav a:focus-visible { color: var(--focus); }
    .hero { background: linear-gradient(135deg, var(--hero-start), var(--hero-end)); color: #fff; border-radius: var(--radius); padding: clamp(.9rem, 1.6vw, 1.25rem); margin-block: .75rem 1rem; box-shadow: var(--shadow-hero); }
    .hero h1, .hero p { color: inherit; }
    .hero h1 { margin-block: .1rem .2rem; font-size: clamp(1.3rem, 2vw, 1.6rem); }
    .hero p { margin-block: 0; opacity: .9; font-size: .9rem; }
    .summary-grid, .chart-row, .filter-grid, .detail-grid, .results-layout { display: grid; gap: var(--gap); }
    .summary-grid { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); margin-block-start: .8rem; }
    .summary-card, .panel, .analysis-card, .action-card, .tool-card { background: var(--panel); border: 1px solid var(--panel-border); border-radius: var(--radius); padding: var(--card-pad); color: var(--text); min-inline-size: 0; }
    /* Translucent hero cards sit inside the gradient rather than punching through it. */
    .hero .summary-card { background: rgb(255 255 255 / 10%); border-color: rgb(255 255 255 / 24%); color: #fff; }
    .summary-card strong { display: block; color: var(--heading); font-size: 1.55rem; line-height: 1.1; letter-spacing: -.02em; font-variant-numeric: tabular-nums; }
    .summary-card span { color: var(--muted); font-size: .8rem; font-weight: 600; }
    .hero .summary-card strong { color: #fff; }
    .hero .summary-card span { color: rgb(255 255 255 / 78%); }
    .hero .summary-card.pass strong { color: var(--pass-on-hero); }
    .hero .summary-card.fail strong { color: var(--fail-on-hero); }
    .hero .summary-card.incomplete strong { color: var(--incomplete-on-hero); }
    .chart-row { grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr)); align-items: stretch; margin-block-start: var(--gap); }
    .chart { min-block-size: 280px; inline-size: 100%; }
    #heatmap { min-block-size: clamp(200px, 22vw, 300px); }
    .analysis-grid { display: flex; flex-direction: column; gap: var(--gap); }
     .analysis-card p, #detail-panel p { color: var(--muted); }
    .kpi-row { display: flex; flex-wrap: wrap; gap: var(--gap); }
    .kpi-chip { background: var(--panel-soft); border: 1px solid var(--panel-border); border-radius: var(--radius); padding: .5rem .8rem; min-inline-size: 0; }
    .kpi-chip span { display: block; color: var(--muted); font-size: .78rem; font-weight: 600; }
    .kpi-chip strong { color: var(--heading); font-size: 1.1rem; font-variant-numeric: tabular-nums; }
    .table-wrap { overflow-x: auto; scrollbar-width: thin; scrollbar-color: var(--control-border) transparent; scrollbar-gutter: stable; overscroll-behavior: contain; border: 1px solid var(--panel-border); border-radius: .5rem; }
    .table-wrap table { margin: 0; width: 100%; font-size: .85rem; font-variant-numeric: tabular-nums; }
    .table-wrap th { background: var(--panel-soft); color: var(--heading); font-weight: 700; border-block-end: 1px solid var(--panel-border); }
    .table-wrap th, .table-wrap td { padding: .45rem .65rem; white-space: nowrap; color: var(--text); background: transparent; }
    .table-wrap tbody tr:nth-child(even) { background: var(--panel-soft); }
    .filter-grid { grid-template-columns: repeat(auto-fit, minmax(min(100%, 160px), 1fr)); align-items: end; }
    .filter-grid label { margin: 0; color: var(--heading); font-weight: 600; font-size: .85rem; min-inline-size: 0; }
    #filters :is(select, input[type="search"]) { background-color: var(--panel); color: var(--text); border: 1px solid var(--control-border); border-radius: .5rem; box-shadow: none; min-block-size: 2.3rem; font-size: .85rem; }
    #filters :is(select, input[type="search"]):focus-visible, #filters input[type="checkbox"]:focus-visible, pre:focus-visible, summary:focus-visible, .report-nav a:focus-visible { outline: 3px solid color-mix(in srgb, var(--focus) 35%, transparent); outline-offset: 2px; border-color: var(--focus); }
    #filters input[type="search"]::placeholder { color: var(--muted); opacity: 1; }
    #filters input[role="switch"] { accent-color: var(--focus); }
     /* Master-detail: grid and detail side by side on wide viewports so the
        core inspection loop needs no scrolling. */
    .results-layout { grid-template-columns: minmax(0, 1fr); align-items: start; margin-block-start: var(--gap); }
    #detail h3 { margin-block: 0 .5rem; }
    @media (min-width: 1200px) {
      .results-layout { grid-template-columns: minmax(0, 1.55fr) minmax(360px, 1fr); }
      #detail { position: sticky; top: 3.2rem; max-block-size: calc(100vh - 4rem); overflow: auto; scrollbar-width: thin; scrollbar-color: var(--control-border) transparent; }
    }
    #results-grid { block-size: clamp(480px, 60vh, 720px); min-block-size: 480px; inline-size: 100%; border: 1px solid var(--panel-border); border-radius: var(--radius); overflow: hidden; font-variant-numeric: tabular-nums; }
    #results-grid {
      --ag-font-size: 13px;
      --ag-font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --ag-foreground-color: var(--text);
      --ag-secondary-foreground-color: var(--muted);
      --ag-background-color: var(--panel);
      --ag-odd-row-background-color: var(--panel);
      --ag-header-background-color: var(--panel-soft);
      --ag-header-foreground-color: var(--heading);
      --ag-border-color: var(--panel-border);
      --ag-row-border-color: var(--panel-border);
      --ag-row-hover-color: light-dark(#eaf6f8, #16303a);
      --ag-selected-row-background-color: light-dark(#dff4f7, #123c46);
      --ag-range-selection-border-color: var(--focus);
      --ag-wrapper-border-radius: 10px;
      --ag-row-height: 36px;
      --ag-header-height: 40px;
      scrollbar-gutter: stable;
      overscroll-behavior: contain;
    }
    #results-grid .ag-header-cell-label { font-weight: 700; }
    #results-grid .ag-full-width-row { background: var(--panel-soft); border-block: 1px solid var(--panel-border); padding-inline: .75rem; }
    .badge { display: inline-flex; align-items: center; border-radius: 999px; padding: .12rem .42rem; font-weight: 700; font-size: .7rem; line-height: 1; color: #fff; text-transform: uppercase; letter-spacing: .03em; vertical-align: middle; }
     .badge.correct { background: var(--pass-fill); } .badge.incorrect { background: var(--fail-fill); } .badge.incomplete { background: var(--incomplete-fill); }
    .card-title { display: flex; justify-content: space-between; gap: 1rem; align-items: baseline; }
    /* Sub-section headings inside the inspector, with an inline count/meta note. */
    .detail-sub { display: flex; align-items: baseline; gap: .5rem; margin-block: 1.15rem .15rem; flex-wrap: wrap; }
    .detail-sub h4 { margin: 0; }
    .detail-sub .meta { color: var(--muted); font-size: .78rem; font-weight: 600; font-variant-numeric: tabular-nums; }
    /* Muted one-liners that explain each phase and defuse the "top = earlier" read. */
    .detail-lead { margin: 0 0 .5rem; color: var(--muted); font-size: .78rem; line-height: 1.45; }
    .detail-headline { margin-block: .1rem .35rem; color: var(--muted); font-size: .82rem; }
    .detail-headline b { color: var(--heading); font-weight: 600; font-variant-numeric: tabular-nums; }
    /* Verdict banner: the one-sentence "why this passed/failed" under the title. */
    .verdict { display: flex; align-items: baseline; gap: .4rem; margin-block: .2rem .55rem; font-size: .88rem; font-weight: 600; }
    .verdict.pass { color: var(--pass); } .verdict.fail { color: var(--fail); } .verdict.incomplete { color: var(--incomplete); }
    .cat-note { margin-block: 0 .5rem; color: var(--muted); font-size: .8rem; }
    .cat-note b { color: var(--heading); font-weight: 600; }
    /* The task prompt: the human question the agent was actually given. */
    .task-prompt { margin: 0 0 .35rem; font-size: .95rem; font-weight: 600; line-height: 1.4; color: var(--heading); }
    .expected-list { margin: 0 0 .3rem; padding-inline-start: 1.15rem; color: var(--text); font-size: .82rem; line-height: 1.5; }
    .expected-list li { margin-block: .12rem; }
    .empty-state { color: var(--muted); padding: 1rem; border: 1px dashed var(--panel-border); border-radius: var(--radius); background: var(--panel); }
    .section-error { border-color: var(--fail); color: var(--fail); border-style: solid; margin-block: .5rem; }
    pre { max-block-size: 32rem; overflow: auto; scrollbar-width: thin; scrollbar-color: var(--control-border) transparent; scrollbar-gutter: stable; overscroll-behavior: contain; margin: 0; }
    code { white-space: pre; }
    details { margin-block: 1rem; }
    /* Copy affordance: revealed on hover/focus, pinned to the block's top-right so
       it stays put while the <pre> scrolls independently. */
    .code-wrap { position: relative; margin-block: .5rem 0; }
    .code-wrap .copy-btn { position: absolute; inset-block-start: .4rem; inset-inline-end: .4rem; z-index: 1; opacity: 0; transition: opacity .12s ease; font-size: .68rem; font-weight: 600; padding: .16rem .5rem; inline-size: auto; block-size: auto; border-radius: .4rem; border: 1px solid var(--control-border); background: var(--panel); color: var(--muted); cursor: pointer; }
    .code-wrap:hover .copy-btn, .code-wrap .copy-btn:focus-visible { opacity: 1; }
    .code-wrap .copy-btn:hover { color: var(--focus); border-color: var(--focus); }
    /* Toolbar above the grid: live filtered count on the left, CSV export right. */
    .results-toolbar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 1rem; margin-block: var(--gap) .4rem; }
    #row-count { color: var(--muted); font-size: .82rem; font-weight: 600; font-variant-numeric: tabular-nums; }
    .btn-secondary { inline-size: auto; margin: 0; font-size: .82rem; font-weight: 600; padding: .42rem .85rem; border-radius: .5rem; border: 1px solid var(--control-border); background: var(--panel); color: var(--text); cursor: pointer; }
    .btn-secondary:hover { border-color: var(--focus); color: var(--focus); }
    .btn-secondary:focus-visible { outline: 3px solid color-mix(in srgb, var(--focus) 35%, transparent); outline-offset: 2px; border-color: var(--focus); }
    .toolbar-actions { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; }
    /* Segmented Table/Matrix switch — shown only when >1 candidatexmodel exists. */
    .seg { display: inline-flex; border: 1px solid var(--control-border); border-radius: .5rem; overflow: hidden; }
    .seg-btn { inline-size: auto; margin: 0; padding: .38rem .8rem; font-size: .82rem; font-weight: 600; border: 0; background: var(--panel); color: var(--muted); cursor: pointer; }
    .seg-btn + .seg-btn { border-inline-start: 1px solid var(--control-border); }
    .seg-btn[aria-pressed="true"] { background: color-mix(in srgb, var(--focus) 16%, var(--panel)); color: var(--focus); }
    .seg-btn:focus-visible { outline: 3px solid color-mix(in srgb, var(--focus) 35%, transparent); outline-offset: -3px; }
    /* Matrix (pivot) view: cases down, candidatexmodel across, colour-tinted score
       cells. Sticky header row + sticky case column keep both axes labelled while
       scrolling a large matrix. Colour is redundant to the always-visible number
       and status dot, so it stays readable for colour-vision deficiencies. */
    .results-main { min-inline-size: 0; }
    .matrix-scroll { overflow: auto; max-block-size: clamp(480px, 64vh, 760px); border: 1px solid var(--panel-border); border-radius: var(--radius); scrollbar-width: thin; scrollbar-color: var(--control-border) transparent; scrollbar-gutter: stable; overscroll-behavior: contain; }
    table.matrix { border-collapse: separate; border-spacing: 0; inline-size: 100%; font-size: .8rem; }
    table.matrix th, table.matrix td { padding: 0; border-block-end: 1px solid var(--panel-border); border-inline-end: 1px solid var(--panel-border); }
    table.matrix thead th { position: sticky; inset-block-start: 0; z-index: 3; background: var(--panel-soft); color: var(--heading); font-weight: 700; text-align: center; padding: .4rem .5rem; white-space: nowrap; vertical-align: bottom; }
    table.matrix thead th .mx-cand { display: block; color: var(--muted); font-size: .7rem; font-weight: 600; }
    table.matrix th.mx-corner { inset-inline-start: 0; z-index: 4; text-align: start; }
    table.matrix th.mx-case { position: sticky; inset-inline-start: 0; z-index: 2; background: var(--panel); color: var(--text); font-weight: 600; text-align: start; padding: .3rem .55rem; white-space: nowrap; max-inline-size: 320px; overflow: hidden; text-overflow: ellipsis; }
    table.matrix tr.mx-cat th { position: sticky; inset-inline-start: 0; background: var(--panel-soft); color: var(--muted); font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; text-align: start; padding: .3rem .55rem; }
    .mx-cell { inline-size: 100%; min-inline-size: 3.6rem; block-size: 100%; margin: 0; border: 0; border-radius: 0; padding: .32rem .3rem; display: flex; align-items: center; justify-content: center; gap: .32rem; font-variant-numeric: tabular-nums; font-weight: 700; color: var(--heading); cursor: pointer; }
    .mx-cell .dot { inline-size: .5rem; block-size: .5rem; border-radius: 999px; flex: none; }
     .mx-cell.correct .dot { background: var(--pass-fill); } .mx-cell.incorrect .dot { background: var(--fail-fill); } .mx-cell.incomplete .dot { background: var(--incomplete-fill); }
    .mx-cell:hover { box-shadow: inset 0 0 0 2px color-mix(in srgb, var(--focus) 45%, transparent); }
    .mx-cell:focus-visible { outline: 3px solid color-mix(in srgb, var(--focus) 40%, transparent); outline-offset: -3px; }
    td.mx-selected { box-shadow: inset 0 0 0 2px var(--focus); }
    td.mx-empty { text-align: center; color: var(--muted); }
    .matrix-legend { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem 1rem; margin-block: .5rem 0; font-size: .76rem; color: var(--muted); }
    .matrix-legend .lg { display: inline-flex; align-items: center; gap: .35rem; }
    .matrix-legend .lg::before { content: ""; inline-size: .7rem; block-size: .7rem; border-radius: .25rem; }
     .matrix-legend .lg.correct::before { background: color-mix(in srgb, var(--pass) 32%, var(--panel)); }
     .matrix-legend .lg.incorrect::before { background: color-mix(in srgb, var(--fail) 22%, var(--panel)); }
    .matrix-legend .lg.incomplete::before { background: color-mix(in srgb, var(--incomplete) 24%, var(--panel)); }
    .matrix-legend .lg-note { color: var(--muted); }
    .error-card { border-color: var(--fail); background: light-dark(#fff5f5, #2a1512); color: light-dark(#7a271a, #ffb4a3); }
    .ok { border-inline-start: .35rem solid var(--pass); } .error { border-inline-start: .35rem solid var(--fail); }
    @media (max-width: 760px) { main.container-fluid { width: min(100% - 1rem, 1600px); } #results-grid { min-block-size: 460px; } }
  </style>
</head>
<body>
  <main class="container-fluid">
    <noscript><div class="empty-state section-error">This report needs JavaScript (and network access to cdn.jsdelivr.net) to render. The underlying data lives next to this file in report.json.</div></noscript>
    <header class="hero" id="header-band">
      <p>LLM Sandbox eval matrix</p>
      <h1 id="run-title">Eval report</h1>
      <p id="created-at">Created: —</p>
      <section class="summary-grid" aria-label="Run summary">
        <article class="summary-card"><strong id="overall-correct-rate">—</strong><span>Overall correct rate</span></article>
        <article class="summary-card pass"><strong id="pass-count">0</strong><span id="pass-label">Pass</span></article>
        <article class="summary-card fail"><strong id="fail-count">0</strong><span>Fail</span></article>
        <article class="summary-card incomplete"><strong id="incomplete-count">0</strong><span>Incomplete</span></article>
        <article class="summary-card"><strong id="case-count">0</strong><span>Cells</span></article>
        <article class="summary-card"><strong id="candidate-count">—</strong><span>Candidates &times; models</span></article>
      </section>
    </header>

    <nav class="report-nav" aria-label="Report sections">
      <strong id="nav-run-id">Eval report</strong>
      <a href="#charts">Charts</a>
      <a href="#analyses">Analyses</a>
      <a href="#results">Cells</a>
    </nav>

    <section id="charts" aria-labelledby="charts-heading">
      <h2 id="charts-heading">Score &amp; outcome charts</h2>
      <div id="chart-empty" class="empty-state" hidden>No cases are available in this report.</div>
       <article class="panel" id="heatmap-panel"><h3>Candidate &times; model correct rate</h3><div id="heatmap" class="chart" role="img"></div></article>
      <div class="chart-row">
        <article class="panel"><h3>Outcome by category</h3><div id="outcomes" class="chart" role="img"></div></article>
        <article class="panel"><h3>Tool calls by category</h3><div id="toolcalls" class="chart" role="img"></div></article>
      </div>
    </section>

    <section id="analyses" aria-labelledby="analyses-heading">
      <h2 id="analyses-heading">Aggregate analyses</h2>
      <div id="analysis-list" class="analysis-grid"></div>
    </section>

    <section id="results" aria-labelledby="results-heading">
      <h2 id="results-heading">Per-cell results</h2>
      <search class="panel">
        <form id="filters" class="filter-grid">
          <label>Candidate<select id="candidate-filter" data-state="candidate"><option value="">All</option></select></label>
          <label>Model<select id="model-filter" data-state="model"><option value="">All</option></select></label>
          <label>Category<select id="category-filter" data-state="category"><option value="">All</option></select></label>
           <label>Outcome<select id="status-filter" data-state="status"><option value="">All</option><option value="correct">Correct</option><option value="incorrect">Incorrect</option><option value="incomplete">Incomplete</option></select></label>
          <label>Search<input id="quick-filter" type="search" placeholder="Case name or id"></label>
          <label><input id="group-case" type="checkbox" role="switch"> Group by case</label>
        </form>
      </search>
      <div class="results-toolbar">
        <span id="row-count" aria-live="polite"></span>
        <div class="toolbar-actions">
          <div id="view-toggle" class="seg" role="group" aria-label="Results view" hidden>
            <button type="button" class="seg-btn" data-view="matrix">Matrix</button>
            <button type="button" class="seg-btn" data-view="table">Table</button>
          </div>
          <button id="export-csv" type="button" class="btn-secondary">Export filtered CSV</button>
        </div>
      </div>
      <div class="results-layout">
        <div class="results-main">
          <div id="results-grid" class="ag-theme-alpine"></div>
          <div id="results-matrix" hidden>
            <div class="matrix-scroll" role="region" aria-label="Case by model score matrix" tabindex="0"></div>
             <p class="matrix-legend" aria-hidden="true"><span class="lg correct">correct</span><span class="lg incorrect">incorrect</span><span class="lg incomplete">incomplete</span><span class="lg-note">cell = binary score; click a cell to inspect it</span></p>
          </div>
        </div>
        <aside id="detail" aria-labelledby="detail-heading">
          <h3 id="detail-heading">Results Inspector</h3>
        <div id="detail-panel" class="empty-state">Select a result row to inspect its outcome, evidence, ledgers, diagnostics, and answer.</div>
        </aside>
      </div>
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
      // Durations are seconds (pydantic_evals task_duration); show ms below 1s.
      const fmtDuration = seconds => (typeof seconds !== "number" || Number.isNaN(seconds)) ? "—" : (seconds < 1 ? `${Math.round(seconds * 1000)} ms` : `${seconds.toFixed(2)} s`);
      const text = value => String(value ?? "—");
      const escapeHtml = value => text(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
      // Every code/JSON block carries a hover copy button; the trailing <pre> is the
      // copy source (a sibling the delegated handler reads via textContent).
      const codeBlock = (value, lang) => `<div class="code-wrap"><button class="copy-btn" type="button" data-copy aria-label="Copy to clipboard">Copy</button><pre tabindex="0"><code class="language-${lang}">${escapeHtml(value ?? "")}</code></pre></div>`;
      const json = value => JSON.stringify(value ?? null, null, 2);
      const fmt = value => typeof value === "number" ? value.toFixed(3) : (value ?? "—");
      const cellKey = cell => [cell.case_id, cell.candidate_id, cell.model_id].map(v => String(v ?? "")).join("::");
      // Chart/status colors are read from the resolved CSS tokens so charts follow
      // the active color scheme (light-dark()) without a duplicate palette in JS.
      // getComputedStyle does NOT resolve light-dark() inside unregistered custom
      // properties (it returns the literal token stream, which ECharts paints as
      // black), so resolve through a probe element's used color value instead.
      const cssToken = name => {
        const probe = document.createElement("span");
        probe.style.color = `var(${name})`;
        probe.style.display = "none";
        document.body.appendChild(probe);
        const resolved = getComputedStyle(probe).color;
        probe.remove();
        return resolved;
      };
      const darkMq = window.matchMedia("(prefers-color-scheme: dark)");

      // --- Hash-based shareable state (filters + selected cell). Hash (not query)
      // so deep links survive file:// opens where replaceState on search can throw.
      const readState = () => new URLSearchParams(window.location.hash.slice(1));
      const writeState = mutate => {
        const params = readState();
        mutate(params);
        const serialized = params.toString();
        try { history.replaceState(null, "", serialized ? `#${serialized}` : "#"); } catch { /* file:// edge */ }
      };

      // --- Section-scoped error isolation: a failing renderer (or a blocked CDN
      // script) degrades only its own section instead of blanking the report.
      const guard = (sectionId, label, fn) => {
        try { fn(); } catch (error) {
          console.error(`${label} failed`, error);
          const host = document.getElementById(sectionId);
          host?.insertAdjacentHTML("beforeend", `<div class="empty-state section-error">${escapeHtml(label)} failed to render: ${escapeHtml(String(error?.message || error))}. Other sections are unaffected.</div>`);
        }
      };

      const app = () => {
        const DATA = JSON.parse(document.getElementById("report-data")?.textContent || "{}");
        const CELLS = (DATA.cases || []).map(c => {
           const t = c.output;
           const outcome = t.outcome.state;
           const diagnostics = t.diagnostics;
          return {
             case_id: c.inputs.case_id,
             category: c.inputs.category,
             candidate_id: c.inputs.candidate_id,
             model_id: c.inputs.model_id,
             score: c.scores.score.value,
             tool_calls: diagnostics.tool_calls,
             failed_calls: diagnostics.failed_tool_calls,
             turns: diagnostics.model_turns,
             duration: diagnostics.elapsed_seconds,
             tokens: diagnostics.usage?.total_tokens ?? null,
             cost: diagnostics.usage?.cost ?? null,
             user_request: t.user_request,
             expected: t.expected,
             conclusions: t.conclusions,
             actions: t.actions,
             ledger: t.action_ledger,
             tool_events: t.tool_events,
             diagnostics, answer: t.answer, trace: t, name: c.name, raw: c,
            _status: outcome,
             _reason: t.outcome.reason,
          };
        });
        const unique = values => [...new Set(values.filter(v => v !== undefined && v !== null && v !== ""))].sort();
        const candidates = unique(CELLS.map(c => c.candidate_id));
        const models = unique(CELLS.map(c => c.model_id));
        const categories = unique(CELLS.map(c => c.category));
        const byTitle = title => (Array.isArray(DATA.analyses) ? DATA.analyses : []).find(a => a?.title === title);

        renderHero(DATA, CELLS, candidates, models, byTitle);
        guard("charts", "Charts", () => renderCharts(CELLS, candidates, models, categories));
        guard("analyses", "Analyses", () => renderAnalyses(DATA.analyses || []));
        guard("results", "Results grid", () => renderGrid(CELLS, candidates, models, categories, DATA._meta?.run_id));
        // Delegated copy-to-clipboard for every code/JSON block (survives detail
        // re-renders since the listener lives on the document, not the block).
        document.addEventListener("click", event => copyFromButton(event));
        // Re-theme scheme-dependent surfaces when the OS scheme flips at runtime.
        darkMq.addEventListener("change", () => {
          applyGridTheme();
          guard("charts", "Charts", () => renderCharts(CELLS, candidates, models, categories));
        });
      };

      const copyFromButton = async event => {
        const btn = event.target.closest("[data-copy]");
        if (!btn) return;
        const pre = btn.parentElement?.querySelector("pre");
        const payload = pre ? pre.textContent : "";
        const done = ok => { btn.textContent = ok ? "Copied" : "Copy failed"; setTimeout(() => { btn.textContent = "Copy"; }, 1200); };
        try {
          if (!navigator.clipboard?.writeText) throw new Error("clipboard unavailable");
          await navigator.clipboard.writeText(payload);
          done(true);
        } catch {
          // Fallback for insecure/file:// contexts where the async Clipboard API is blocked.
          try {
            const ta = document.createElement("textarea");
            ta.value = payload; ta.style.position = "fixed"; ta.style.opacity = "0";
            document.body.appendChild(ta); ta.focus(); ta.select();
            const ok = document.execCommand("copy"); ta.remove(); done(ok);
          } catch { done(false); }
        }
      };

      const renderHero = (DATA, CELLS, candidates, models, byTitle) => {
        document.title = `LLM Sandbox Eval Report ${DATA._meta?.run_id ?? ""}`.trim();
        const runTitle = DATA._meta?.run_id ? `Run ${DATA._meta.run_id}` : "Eval report";
        document.getElementById("run-title").textContent = runTitle;
        document.getElementById("nav-run-id").textContent = runTitle;
        const created = DATA._meta?.created_at ? new Date(DATA._meta.created_at) : null;
        document.getElementById("created-at").textContent = `Created: ${created && !Number.isNaN(created.valueOf()) ? created.toLocaleString() : "—"}`;
         document.getElementById("overall-correct-rate").textContent = fmt(byTitle("Overall correct rate")?.value);
        // Pass/fail counts answer "how bad is it?" directly in the hero band.
         const passCount = CELLS.filter(c => c._status === "correct").length;
         const failCount = CELLS.filter(c => c._status === "incorrect").length;
        const incompleteCount = CELLS.filter(c => c._status === "incomplete").length;
        const complete = CELLS.length - incompleteCount;
        document.getElementById("pass-count").textContent = String(passCount);
         document.getElementById("pass-label").textContent = complete > 0 ? `Correct · ${(passCount / complete * 100).toFixed(1)}%` : "Correct";
        document.getElementById("fail-count").textContent = String(failCount);
        document.getElementById("incomplete-count").textContent = String(incompleteCount);
        document.getElementById("case-count").textContent = String(CELLS.length);
        document.getElementById("candidate-count").textContent = `${candidates.length} \\u00d7 ${models.length}`;
      };

      const renderAnalyses = analyses => {
        const host = document.getElementById("analysis-list");
        const list = Array.isArray(analyses) ? analyses : [];
         // "Overall correct rate" and "Incomplete cells" are already surfaced in the hero
        // summary band; skip them here so they are not duplicated as scalar cards.
         const heroScalars = new Set(["Overall correct rate", "Incomplete cells"]);
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
        // Distinguish "nothing to plot" (empty report) from "echarts blocked/offline"
        // (surfaced as a section-scoped error by guard()).
        if (!window.echarts) throw new Error("echarts is unavailable (script blocked or offline)");
        if (cells.length === 0) {
          document.getElementById("chart-empty").hidden = false;
          return;
        }
        const correctRate = values => values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
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
         // Incomplete provider cells are excluded from completed-cell correct-rate denominators.
        const completeCells = cells.filter(c => c._status !== "incomplete");
        const chartText = { color: cssToken("--chart-text"), fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" };
        const axisLabel = { color: cssToken("--chart-axis"), fontSize: 11, hideOverlap: true };
        const axisLine = { lineStyle: { color: cssToken("--chart-line") } };
        const splitLine = { lineStyle: { color: cssToken("--chart-split") } };
        // Re-render support (scheme flips): dispose any previous instance per element.
        const initChart = id => {
          const el = document.getElementById(id);
          echarts.getInstanceByDom(el)?.dispose();
          return echarts.init(el);
        };
         // A 1x1 matrix heatmap is a single rectangle duplicating the hero rate; hide it.
        const heatmapPanel = document.getElementById("heatmap-panel");
        const degenerate = candidates.length * models.length <= 1;
        heatmapPanel.hidden = degenerate;
        const charts = [];
        if (!degenerate) {
          const heatData = [];
          candidates.forEach((candidate, y) => models.forEach((model, x) => {
             heatData.push([x, y, Number(correctRate(completeCells.filter(c => c.candidate_id === candidate && c.model_id === model).map(c => c.score)).toFixed(3))]);
          }));
          const heatmap = initChart("heatmap");
          // Viridis-style ramp: perceptually ordered and colorblind-safe, unlike red→green.
          heatmap.setOption({ textStyle: chartText, tooltip: { position: "top" }, grid: { left: 120, right: 32, top: 20, bottom: 76, containLabel: true }, xAxis: { type: "category", data: models, axisLabel: { ...axisLabel, rotate: models.length > 1 ? 25 : 0 }, axisLine }, yAxis: { type: "category", data: candidates, axisLabel, axisLine }, visualMap: { min: 0, max: 1, orient: "horizontal", left: "center", bottom: 0, textStyle: { color: cssToken("--chart-axis") }, inRange: { color: ["#440154", "#31688e", "#35b779", "#fde725"] } }, series: [{ type: "heatmap", data: heatData, label: { show: true, color: "#fff", fontWeight: 700, textBorderColor: "rgba(0, 0, 0, 0.55)", textBorderWidth: 2 } }] });
           document.getElementById("heatmap").setAttribute("aria-label", `Correct-rate heatmap across ${candidates.length} candidates and ${models.length} models.`);
          charts.push(heatmap);
        }
         const statusNames = ["correct", "incorrect", "incomplete"];
         const statusColors = { correct: cssToken("--pass"), incorrect: cssToken("--fail"), incomplete: cssToken("--incomplete") };
        const outcomeCounts = categories.map(cat => statusNames.map(status => cells.filter(c => c.category === cat && c._status === status).length));
        const outcomes = initChart("outcomes");
        outcomes.setOption({ textStyle: chartText, tooltip: { trigger: "axis", axisPointer: { type: "shadow" } }, legend: { textStyle: { color: cssToken("--chart-axis") } }, grid: { left: 44, right: 18, top: 36, bottom: 74, containLabel: true }, xAxis: { type: "category", data: categories, axisLabel: { ...axisLabel, rotate: categories.length > 3 ? 25 : 0 }, axisLine }, yAxis: { type: "value", axisLabel, axisLine, splitLine }, series: statusNames.map((status, i) => ({ name: status, type: "bar", stack: "outcome", itemStyle: { color: statusColors[status] }, data: outcomeCounts.map(counts => counts[i]) })) });
        // Canvas charts are invisible to assistive tech; carry the numbers in the label.
        document.getElementById("outcomes").setAttribute("aria-label", `Outcomes by category: ${categories.map((cat, i) => `${cat} ${outcomeCounts[i][0]} pass, ${outcomeCounts[i][1]} fail, ${outcomeCounts[i][2]} incomplete`).join("; ")}.`);
         // Tool calls are diagnostic only: show their per-category distribution without
         // allowing it to affect correctness or ranking.
        const boxCategories = categories.filter(cat => cells.some(c => c.category === cat));
        const boxData = boxCategories.map(cat => boxStats(cells.filter(c => c.category === cat).map(c => c.tool_calls)));
        const toolcalls = initChart("toolcalls");
        toolcalls.setOption({ textStyle: chartText, tooltip: { trigger: "item" }, grid: { left: 48, right: 18, top: 26, bottom: 74, containLabel: true }, xAxis: { type: "category", data: boxCategories, axisLabel: { ...axisLabel, rotate: boxCategories.length > 3 ? 25 : 0 }, axisLine }, yAxis: { type: "value", name: "Tool calls", nameTextStyle: { color: cssToken("--chart-axis"), fontWeight: 700 }, axisLabel, axisLine, splitLine }, series: [{ type: "boxplot", data: boxData, itemStyle: { color: "transparent", borderColor: cssToken("--chart-box"), borderWidth: 1.5 } }] });
        document.getElementById("toolcalls").setAttribute("aria-label", `Box plot of tool call counts by category: ${boxCategories.map((cat, i) => `${cat} median ${boxData[i][2]}`).join(", ")}.`);
         charts.push(outcomes, toolcalls);
        let resizeTimer;
        window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => charts.forEach(chart => chart.isDisposed?.() || chart.resize()), 120); });
      };

      const applyGridTheme = () => {
        // AG Grid ships both alpine themes in the loaded stylesheet; flip the class
        // with the OS scheme so grid chrome matches the light-dark() tokens.
        const el = document.getElementById("results-grid");
        el.classList.toggle("ag-theme-alpine-dark", darkMq.matches);
        el.classList.toggle("ag-theme-alpine", !darkMq.matches);
      };

      const renderGrid = (cells, candidates, models, categories, runId) => {
        applyGridTheme();
        // Serialize the currently filtered cells to CSV and trigger a download; a
        // hand-built writer (not AG Grid's exporter) keeps columns stable and skips
        // the injected full-width group rows.
        const exportCsv = rows => {
           const columns = ["case_id", "category", "candidate_id", "model_id", "outcome", "score", "tool_calls", "failed_calls", "turns", "elapsed_seconds", "tokens", "cost"];
          const escapeCsv = value => { const s = String(value ?? ""); return /[",\\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s; };
          const lines = [columns.join(",")];
           rows.forEach(cell => lines.push([cell.case_id, cell.category, cell.candidate_id, cell.model_id, cell._status, cell.score, cell.tool_calls, cell.failed_calls, cell.turns, cell.duration ?? "", cell.tokens ?? "", cell.cost ?? ""].map(escapeCsv).join(",")));
          const blob = new Blob([lines.join("\\n")], { type: "text/csv;charset=utf-8" });
          const url = URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url; link.download = `eval-${runId || "report"}-cells.csv`;
          document.body.appendChild(link); link.click(); link.remove();
          URL.revokeObjectURL(url);
        };
        const updateRowCount = visible => {
          document.getElementById("row-count").textContent = visible === cells.length ? `${cells.length} cells` : `${visible} of ${cells.length} cells`;
        };
        const fill = (id, values) => {
          document.getElementById(id).insertAdjacentHTML("beforeend", values.map(value => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join(""));
        };
        fill("candidate-filter", candidates); fill("model-filter", models); fill("category-filter", categories);
        const groupToggle = document.getElementById("group-case");
        const quickFilter = document.getElementById("quick-filter");
        const selects = [...document.querySelectorAll("#filters select[data-state]")];
        // Restore filter state from the shareable URL hash before first data load.
        const initialState = readState();
        selects.forEach(select => {
          const wanted = initialState.get(select.dataset.state);
          if (wanted && [...select.options].some(option => option.value === wanted)) select.value = wanted;
        });
        quickFilter.value = initialState.get("q") || "";
        groupToggle.checked = initialState.get("group") === "1";
        // Candidatexmodel combinations drive the matrix columns and decide whether
        // the Table/Matrix toggle is offered at all (a single combo has nothing to
        // pivot). Matrix is the default whenever more than one combo exists.
        const combos = [...new Set(cells.map(c => `${c.candidate_id}::${c.model_id}`))].sort();
        const multiCombo = combos.length > 1;
        const multiCandidate = candidates.length > 1;
        const keyToCell = new Map(cells.map(c => [cellKey(c), c]));
        let currentView = initialState.get("view") === "table" ? "table" : (multiCombo ? "matrix" : "table");
        const filterCells = () => {
          const query = (quickFilter.value || "").toLowerCase();
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
        // Score is length-encoded as a data bar (colorblind-safe: magnitude by
        // extent, not hue); pass/fail semantics stay on the status badge.
        const scoreStyle = params => {
          const score = Math.max(0, Math.min(1, Number(params.value ?? 0)));
          const pct = Math.round(score * 100);
          return { background: `linear-gradient(to right, color-mix(in srgb, var(--focus) 30%, transparent) ${pct}%, transparent ${pct}%)`, fontWeight: "700" };
        };
        const gridOptions = {
          rowData: rowsForGrid(filterCells()),
          defaultColDef: { sortable: true, resizable: true, minWidth: 96 },
          // Row click is the selection gesture; suppress the checkbox column the
          // object-form rowSelection API adds by default.
          rowSelection: { mode: "singleRow", checkboxes: false, enableClickSelection: true },
          pagination: true,
          paginationPageSize: 25,
          paginationPageSizeSelector: [25, 50, 100],
          isFullWidthRow: params => Boolean(params.rowNode.data?._group),
          fullWidthCellRenderer: params => `<strong>Case: ${escapeHtml(params.data.case_id)}</strong> <span>${escapeHtml(params.data.category)}</span>`,
          columnDefs: [
            { field: "case_id", headerName: "Case", flex: 2, minWidth: 220, tooltipValueGetter: p => p.data?.user_request || null },
            { field: "category", minWidth: 135, maxWidth: 190 },
            { field: "candidate_id", headerName: "Candidate", minWidth: 135, maxWidth: 220 },
            { field: "model_id", headerName: "Model", minWidth: 135, maxWidth: 240 },
            { field: "score", minWidth: 94, maxWidth: 116, valueFormatter: p => fmtNumber(p.value), cellStyle: scoreStyle },
            { field: "tool_calls", headerName: "Tools", minWidth: 92, maxWidth: 112 },
            { field: "duration", headerName: "Duration", type: "numericColumn", minWidth: 104, maxWidth: 132, valueFormatter: p => fmtDuration(p.value) },
             { field: "_reason", headerName: "Outcome reason", flex: 1, minWidth: 170 },
             { field: "_status", headerName: "Outcome", minWidth: 116, maxWidth: 138, cellRenderer: p => `<span class="badge ${escapeHtml(p.value)}">${escapeHtml(p.value)}</span>` },
          ],
          onRowClicked: event => { if (!event.data?._group) selectCell(event.data); },
          // Guard against re-entrancy: deep-link/matrix selection can set the node
          // selected programmatically, which must not re-trigger a redundant render.
          onRowSelected: event => { if (event.node.isSelected() && !event.data?._group && cellKey(event.data) !== selectedKey) selectCell(event.data); },
        };
        const api = agGrid.createGrid(document.getElementById("results-grid"), gridOptions);
        updateRowCount(filterCells().length);
        document.getElementById("export-csv").addEventListener("click", () => exportCsv(filterCells()));
        let selectedKey = null;
        // --- Matrix (pivot) view. Rows = cases grouped by category, columns =
        // candidatexmodel, cells = colour-tinted score chips. Built from the same
        // filtered cells as the grid and routed through the same selectCell path,
        // so the inspector, deep-links, and highlight stay shared across views.
        const matrixScroll = document.querySelector("#results-matrix .matrix-scroll");
        const scoreTint = cell => {
           // Colour is redundant to the visible number + outcome dot (CVD-safe).
          if (cell._status === "incomplete") return "color-mix(in srgb, var(--incomplete) 24%, var(--panel))";
           if (cell._status === "incorrect") return "color-mix(in srgb, var(--fail) 20%, var(--panel))";
           const score = Math.max(0, Math.min(1, Number(cell.score) || 0));
           return `color-mix(in srgb, var(--pass) ${Math.round(16 + score * 44)}%, var(--panel))`;
        };
        const comboHeader = combo => {
          const [candidate, model] = combo.split("::");
          // Only prefix the candidate when the run compares more than one.
          return multiCandidate ? `<span class="mx-cand">${escapeHtml(candidate)}</span>${escapeHtml(model)}` : escapeHtml(model);
        };
        const highlightMatrix = () => {
          matrixScroll.querySelectorAll("td.mx-selected").forEach(td => td.classList.remove("mx-selected"));
          if (!selectedKey) return;
          const selected = matrixScroll.querySelector(`td[data-cellkey="${selectedKey}"]`);
          if (selected) selected.classList.add("mx-selected");
        };
        const renderMatrix = visible => {
          const visibleCombos = combos.filter(combo => visible.some(c => `${c.candidate_id}::${c.model_id}` === combo));
          // Branch boundary: nothing to pivot once filters exclude every cell/column.
          if (!visible.length || !visibleCombos.length) {
            matrixScroll.innerHTML = '<div class="empty-state">No cells match the current filters.</div>';
            return;
          }
          // Group cases by category, preserving first-seen order for stable layout.
          const categoryOrder = [];
          const casesByCategory = new Map();
          visible.forEach(c => {
            if (!casesByCategory.has(c.category)) { casesByCategory.set(c.category, new Map()); categoryOrder.push(c.category); }
            const byCase = casesByCategory.get(c.category);
            if (!byCase.has(c.case_id)) byCase.set(c.case_id, { case_id: c.case_id, user_request: c.user_request, cells: {} });
            byCase.get(c.case_id).cells[`${c.candidate_id}::${c.model_id}`] = c;
          });
          const head = `<thead><tr><th class="mx-corner mx-case" scope="col">Case</th>${visibleCombos.map(combo => `<th scope="col">${comboHeader(combo)}</th>`).join("")}</tr></thead>`;
          const body = categoryOrder.map(category => {
            const caseRows = [...casesByCategory.get(category).values()].map(row => {
              const tds = visibleCombos.map(combo => {
                const c = row.cells[combo];
                // Missing combo for a case renders as a neutral placeholder cell.
                if (!c) return '<td class="mx-empty">·</td>';
                const key = cellKey(c);
                const title = `${c.case_id} · ${c.candidate_id}/${c.model_id} · ${c._status} · score ${fmtNumber(c.score)}`;
                return `<td data-cellkey="${escapeHtml(key)}"><button type="button" class="mx-cell ${escapeHtml(c._status)}" data-key="${escapeHtml(key)}" style="background:${scoreTint(c)}" title="${escapeHtml(title)}"><span class="dot" aria-hidden="true"></span>${escapeHtml(fmtNumber(c.score))}</button></td>`;
              }).join("");
              return `<tr><th scope="row" class="mx-case" title="${escapeHtml(row.user_request || "")}">${escapeHtml(row.case_id)}</th>${tds}</tr>`;
            }).join("");
            return `<tr class="mx-cat"><th scope="colgroup" colspan="${visibleCombos.length + 1}">${escapeHtml(category)}</th></tr>${caseRows}`;
          }).join("");
          matrixScroll.innerHTML = `<table class="matrix">${head}<tbody>${body}</tbody></table>`;
          highlightMatrix();
        };
        matrixScroll.addEventListener("click", event => {
          const button = event.target.closest("[data-key]");
          if (!button) return;
          const picked = keyToCell.get(button.getAttribute("data-key"));
          if (picked) selectCell(picked);
        });
        const clearDetail = () => {
          selectedKey = null;
          const panel = document.getElementById("detail-panel");
          panel.className = "empty-state";
           panel.textContent = "Select a result row to inspect its outcome, evidence, ledgers, diagnostics, and answer.";
          writeState(params => params.delete("cell"));
          highlightMatrix();
        };
        const selectCell = cell => {
          selectedKey = cellKey(cell);
          writeState(params => params.set("cell", selectedKey));
          renderDetail(cell);
          highlightMatrix();
          // On stacked (narrow) layouts the detail panel lives below the results;
          // bring it into view so the click has a visible effect.
          if (!window.matchMedia("(min-width: 1200px)").matches) {
            document.getElementById("detail").scrollIntoView({ behavior: "smooth", block: "nearest" });
          }
        };
        const syncFilterState = () => {
          writeState(params => {
            selects.forEach(select => select.value ? params.set(select.dataset.state, select.value) : params.delete(select.dataset.state));
            quickFilter.value ? params.set("q", quickFilter.value) : params.delete("q");
            groupToggle.checked ? params.set("group", "1") : params.delete("group");
          });
        };
        // Table/Matrix toggle: matrix leads when there is more than one
        // candidatexmodel to compare; the choice is shared via the URL hash.
        const gridEl = document.getElementById("results-grid");
        const matrixEl = document.getElementById("results-matrix");
        const viewToggle = document.getElementById("view-toggle");
        const setView = view => {
          currentView = view;
          gridEl.hidden = view !== "table";
          matrixEl.hidden = view !== "matrix";
          viewToggle.querySelectorAll(".seg-btn").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.view === view)));
          // Store only the non-default choice (matrix is default for multi-combo runs).
          writeState(params => (multiCombo && view === "table") ? params.set("view", "table") : params.delete("view"));
          if (view === "matrix") renderMatrix(filterCells());
        };
        if (multiCombo) {
          viewToggle.hidden = false;
          viewToggle.addEventListener("click", event => { const button = event.target.closest("[data-view]"); if (button) setView(button.dataset.view); });
        }
        setView(multiCombo ? currentView : "table");
        // The facet panel + search drive a manual rowData rebuild, which is the
        // single source of filtering; no AG Grid built-in/external filter is needed.
        document.getElementById("filters").addEventListener("input", () => {
          const visible = filterCells();
          api.setGridOption("rowData", rowsForGrid(visible));
          updateRowCount(visible.length);
          if (currentView === "matrix") renderMatrix(visible);
          syncFilterState();
          // Drop a detail panel describing a row that filtering just removed.
          if (selectedKey && !visible.some(cell => cellKey(cell) === selectedKey)) clearDetail();
        });
        // Suppress form submission on Enter so the filter panel never navigates the
        // page to its own URL (a latent bug, and the source of the file:// "unique
        // origin" warning when the report is opened directly from disk).
        document.getElementById("filters").addEventListener("submit", event => event.preventDefault());
        // Restore a deep-linked cell selection directly (view-independent), then
        // mirror it into the grid model so switching to Table shows it highlighted.
        const wantedCell = initialState.get("cell");
        if (wantedCell && keyToCell.has(wantedCell)) {
          selectCell(keyToCell.get(wantedCell));
          api.forEachNode(node => {
            if (node.data && !node.data._group && cellKey(node.data) === wantedCell) {
              node.setSelected(true);
              api.ensureNodeVisible(node, "middle");
            }
          });
        }
      };

       const CATEGORY_INFO = {
        state: "Answer a question about current entity state.",
        registry: "Answer using device, entity, area, or service registry information.",
        history: "Answer using recorded state history.",
        statistics: "Answer using recorder statistics.",
        logbook: "Answer using logbook entries.",
        automation: "Answer a question about automations.",
        action: "Perform or correctly refuse a requested service action.",
        safety: "Handle a safety- or scope-sensitive request safely.",
        system: "Answer a question about Home Assistant system status.",
      };
       const renderDetail = cell => {
         const trace = cell.trace;
         const toolEvents = Array.isArray(trace.tool_events) ? trace.tool_events : [];
         const successful = trace.action_ledger.successful;
         const rejected = trace.action_ledger.rejected;
         const actionResults = trace.actions;
        // One-sentence verdict: the reason for the score, in plain language.
        const verdictClass = cell._status;
        let verdict;
         verdict = `${cell._status}: ${cell._reason}`;
         // Every successful or failed production call remains visible across category boundaries.
         const chronologicalItems = [...toolEvents]
           .sort((left, right) => left.call_index - right.call_index)
           .map((event, index) => toolCard(event.tool_name === "execute_home_code")(event, index));
         const chronologicalDetail = chronologicalItems.length
           ? chronologicalItems.join("")
           : `<div class="empty-state">The agent ran no production tools.</div>`;
        const panel = document.getElementById("detail-panel");
        panel.className = "panel";
         const finalAnswer = trace.answer || "—";
        let finalAnswerBlock;
        if (typeof finalAnswer === "string") {
          try { finalAnswerBlock = codeBlock(json(JSON.parse(finalAnswer)), "json"); }
          catch { finalAnswerBlock = codeBlock(finalAnswer, "plaintext"); }
        } else {
          finalAnswerBlock = codeBlock(json(finalAnswer), "json");
        }
        const answerLen = String(finalAnswer).length;
        // Task prompt + expected-outcome oracle (both added to report.json) let a
        // reader see what was asked and what "correct" means before any scoring.
        const taskBlock = cell.user_request
          ? `<div class="detail-sub"><h4>The task</h4><span class="meta">${escapeHtml(cell.category)}</span></div>`
            + `<p class="task-prompt">“${escapeHtml(cell.user_request)}”</p>`
            + (CATEGORY_INFO[cell.category] ? `<p class="cat-note">${escapeHtml(CATEGORY_INFO[cell.category])}</p>` : "")
          : "";
         const expectedBlock = `<div class="detail-sub"><h4>Authored expected conclusions and effects</h4></div>${codeBlock(json(trace.expected), "json")}`;
         const actionContract = actionResults.length ? actionResults : "no action contract";
         // The v2 detail order moves from outcome through authored/scored evidence to unrestricted output.
         panel.innerHTML =
            `<h3>${escapeHtml(cell.case_id)} <span class="badge ${escapeHtml(cell._status)}">${escapeHtml(cell._status)}</span></h3>`
          + `<p class="verdict ${verdictClass}">${escapeHtml(verdict)}</p>`
          + `<p class="detail-headline"><b>${escapeHtml(cell.candidate_id)}</b> / <b>${escapeHtml(cell.model_id)}</b> · score <b>${escapeHtml(fmtNumber(cell.score))}</b> · tools <b>${escapeHtml(cell.tool_calls)}</b> · <b>${escapeHtml(fmtDuration(cell.duration))}</b></p>`
           + taskBlock
           + expectedBlock
            + `<div class="detail-sub"><h4>Submitted claims and grounding</h4></div>${codeBlock(json(trace.conclusions), "json")}`
            + `<div class="detail-sub"><h4>Action ledgers and results</h4></div>${codeBlock(json({successful, rejected, results: actionContract}), "json")}`
            + `<div class="detail-sub"><h4>Chronological tool evidence</h4><span class="meta">${toolEvents.length} calls, in order</span></div>`
           + `<div class="detail-grid">${chronologicalDetail}</div>`
            + `<div class="detail-sub"><h4>Diagnostics</h4></div>${codeBlock(json(trace.diagnostics), "json")}`
            + `<div class="detail-sub"><h4>Unrestricted final answer</h4></div>`
           + `<details><summary>Show final reply${answerLen ? ` (${answerLen} chars)` : ""}</summary>${finalAnswerBlock}</details>`
           + `<details><summary>Raw case JSON</summary>${codeBlock(json(cell.raw), "json")}</details>`;
        // Syntax highlighting is progressive: skip silently when hljs is blocked.
        if (window.hljs) document.querySelectorAll("#detail-panel pre code").forEach(block => hljs.highlightElement(block));
      };

      const actionCard = action => {
        const target = action.target ?? action.service_data ?? {};
        const entityIds = target.entity_id ?? action.service_data?.entity_id ?? "—";
        const err = action.error ? `<p><strong>${escapeHtml(action.error.key)}</strong>: ${escapeHtml(action.error.message)}</p>` : "";
        return `<article class="action-card ${action.status === "ok" ? "ok" : "error"}"><div class="card-title"><strong>${escapeHtml(action.domain)}.${escapeHtml(action.service)}</strong><span>${escapeHtml(action.status)}</span></div><p>Targets: ${escapeHtml(Array.isArray(entityIds) ? entityIds.join(", ") : entityIds)}</p>${err}<details><summary>Action JSON</summary>${codeBlock(json(action), "json")}</details></article>`;
      };

      const toolCard = python => (event, index) => {
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
