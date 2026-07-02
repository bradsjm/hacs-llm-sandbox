"""Artifact writing and leaderboard rendering for eval runs.

The ``report`` CLI command reloads ``run.json`` only. That file stores enough
metadata plus serialized ``CandidateModelScore`` objects to render the same
leaderboard without loading full prompts or per-case traces.
"""

import json
import re
from dataclasses import asdict
from pathlib import Path

from llm_sandbox_evals.schema import CandidateModelScore, CaseTrace, RunResult

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]")
_INTEGRATION_MANIFEST = Path(__file__).resolve().parent.parent / "custom_components" / "llm_sandbox" / "manifest.json"
# Two candidate means within this epsilon are treated as equal quality and tie-broken by smaller prompt size.
_SIZE_TIE_EPSILON = 0.005


def write_run(result: RunResult, runs_dir: Path) -> Path:
    """Write run artifacts and return the created run directory."""
    run_dir = runs_dir / result.run_id
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=False)

    run_json = _run_json(result)
    (run_dir / "run.json").write_text(json.dumps(run_json, indent=2) + "\n", encoding="utf-8")
    (run_dir / "leaderboard.md").write_text(render_leaderboard(result), encoding="utf-8")

    result_lines: list[dict[str, object]] = []
    failure_lines: list[dict[str, object]] = []
    for trace in result.traces:
        result_line = _result_line(trace)
        result_lines.append(result_line)
        # Branch boundary: failures are zero scores or any failed required check.
        if trace.score == 0.0 or any(check.required and not check.passed for check in trace.checks):
            failure_lines.append(result_line)
        trace_path = traces_dir / _trace_filename(trace)
        trace_path.write_text(json.dumps(_trace_json(trace), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _write_jsonl(run_dir / "results.jsonl", result_lines)
    _write_jsonl(run_dir / "failures.jsonl", failure_lines)
    return run_dir


def render_leaderboard(result: RunResult) -> str:
    """Render a plain-Markdown leaderboard from a full run result."""
    return render_leaderboard_from_scores(
        scores=result.scores,
        run_id=result.run_id,
        created_at=result.created_at,
        case_count=len(result.case_ids),
        candidate_ids=result.candidate_ids,
        model_ids=result.model_ids,
    )


def render_leaderboard_from_scores(
    *,
    scores: list[CandidateModelScore],
    run_id: str,
    created_at: str,
    case_count: int,
    candidate_ids: list[str],
    model_ids: list[str],
) -> str:
    """Render a leaderboard from serialized score summaries in ``run.json``."""
    categories = _categories(scores)
    lines = [f"# Eval leaderboard {run_id}", "", f"Created: {created_at}", f"Cases: {case_count}", ""]
    lines.extend(_candidate_table(scores, candidate_ids, model_ids, categories))
    lines.extend(("", "## Candidate x model means", ""))
    lines.extend(_model_matrix_table(scores, candidate_ids, model_ids))
    return "\n".join(lines) + "\n"


def load_results(results_jsonl: Path) -> list[dict[str, object]]:
    """Load ``results.jsonl`` rows for simple downstream inspection."""
    rows: list[dict[str, object]] = []
    for line in results_jsonl.read_text(encoding="utf-8").splitlines():
        # Branch boundary: tolerate blank lines in manually edited/copied jsonl files.
        if not line.strip():
            continue
        decoded = json.loads(line)
        if isinstance(decoded, dict):
            rows.append(dict(decoded))
    return rows


def load_run_json(run_json: Path) -> tuple[str, str, int, list[str], list[str], list[CandidateModelScore]]:
    """Load leaderboard metadata and score summaries from ``run.json``."""
    decoded = json.loads(run_json.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("run.json must contain a JSON object")

    return (
        _string_field(decoded, "run_id"),
        _string_field(decoded, "created_at"),
        _int_field(decoded, "case_count"),
        _string_list(decoded.get("candidate_ids")),
        _string_list(decoded.get("model_ids")),
        _scores_field(decoded.get("scores")),
    )


def _run_json(result: RunResult) -> dict[str, object]:
    """Build metadata-only run.json content; prompts stay in trace files."""
    return {
        "run_id": result.run_id,
        "created_at": result.created_at,
        "integration_version": _integration_version(),
        "candidate_ids": result.candidate_ids,
        "model_ids": result.model_ids,
        "case_ids": result.case_ids,
        "case_count": len(result.case_ids),
        "scores": [asdict(score) for score in result.scores],
    }


def _integration_version() -> str:
    """Read the integration version from the checked-in Home Assistant manifest."""
    decoded = json.loads(_INTEGRATION_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("manifest.json must contain a JSON object")

    version = decoded.get("version")
    if not isinstance(version, str):
        raise ValueError("manifest.json field 'version' must be a string")
    return version


def _result_line(trace: CaseTrace) -> dict[str, object]:
    """Build one compact results/failures JSONL row."""
    return {
        "case_id": trace.case_id,
        "category": trace.category,
        "candidate_id": trace.candidate_id,
        "model_id": trace.model_id,
        "score": trace.score,
        "turns": trace.turns,
        "par_turns": trace.par_turns,
        "checks": [{"name": check.name, "passed": check.passed, "required": check.required} for check in trace.checks],
    }


def _trace_json(trace: CaseTrace) -> dict[str, object]:
    """Build one full per-trace artifact."""
    return {
        "case_id": trace.case_id,
        "category": trace.category,
        "candidate_id": trace.candidate_id,
        "model_id": trace.model_id,
        "score": trace.score,
        "turns": trace.turns,
        "par_turns": trace.par_turns,
        "final_answer": trace.final_answer,
        "prompt": trace.prompt,
        "raw_output": trace.raw_output,
        "tool_call": trace.tool_call,
        "tool_result": trace.tool_result,
        "steps": [asdict(step) for step in trace.steps],
        "recorded_actions": list(trace.recorded_actions),
        "checks": [asdict(check) for check in trace.checks],
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write JSONL rows with deterministic key order."""
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _trace_filename(trace: CaseTrace) -> str:
    """Return the filesystem-safe trace filename for one matrix cell."""
    parts = (_sanitize(trace.case_id), _sanitize(trace.model_id), _sanitize(trace.candidate_id))
    return f"{parts[0]}.{parts[1]}.{parts[2]}.json"


def _sanitize(value: str) -> str:
    """Replace unsafe filename characters with underscores."""
    return _SAFE_FILENAME_RE.sub("_", value)


def _categories(scores: list[CandidateModelScore]) -> list[str]:
    """Return categories present in score summaries, preserving first-seen order."""
    categories: list[str] = []
    for score in scores:
        for category in score.per_category:
            if category not in categories:
                categories.append(category)
    return categories


def _candidate_table(
    scores: list[CandidateModelScore],
    candidate_ids: list[str],
    model_ids: list[str],
    categories: list[str],
) -> list[str]:
    """Render candidate ranking rows."""
    score_map = {(score.candidate_id, score.model_id): score for score in scores}
    ranked_rows: list[tuple[str, float, float, float, dict[str, float], int, int, float]] = []
    candidate_prompt_chars: dict[str, int] = {}
    for candidate_id in candidate_ids:
        candidate_scores = [
            score_map[(candidate_id, model_id)] for model_id in model_ids if (candidate_id, model_id) in score_map
        ]
        prompt_chars = candidate_scores[0].prompt_chars if candidate_scores else 0
        candidate_prompt_chars[candidate_id] = prompt_chars

    baseline_prompt_chars = candidate_prompt_chars.get("baseline", max(candidate_prompt_chars.values(), default=0))
    if baseline_prompt_chars == 0:
        baseline_prompt_chars = 1

    for candidate_id in candidate_ids:
        candidate_scores = [
            score_map[(candidate_id, model_id)] for model_id in model_ids if (candidate_id, model_id) in score_map
        ]
        prompt_chars = candidate_prompt_chars[candidate_id]
        api_prompt_chars = candidate_scores[0].api_prompt_chars if candidate_scores else 0
        size_ratio = prompt_chars / baseline_prompt_chars if baseline_prompt_chars else 0.0
        all_case_scores = [case_score for score in candidate_scores for case_score in score.case_scores.values()]
        model_means = [score.mean for score in candidate_scores]
        turns = [score.mean_turns for score in candidate_scores]
        category_means: dict[str, float] = {}
        for category in categories:
            category_values = [
                score.per_category[category] for score in candidate_scores if category in score.per_category
            ]
            category_means[category] = _mean(category_values)
        ranked_rows.append(
            (
                candidate_id,
                _mean(all_case_scores),
                min(model_means, default=0.0),
                _mean(turns),
                category_means,
                prompt_chars,
                api_prompt_chars,
                size_ratio,
            )
        )

    ranked_rows.sort(
        key=lambda row: (
            -round(row[1] / _SIZE_TIE_EPSILON),
            row[6],
            -row[2],
        )
    )
    header = ["Candidate", "Mean", "MinModel", "Turns", "PromptChars", "SizeRatio", *categories]
    lines = [_markdown_row(header), _markdown_separator(len(header))]
    for (
        candidate_id,
        mean,
        min_model,
        mean_turns,
        category_means,
        prompt_chars,
        _api_prompt_chars,
        size_ratio,
    ) in ranked_rows:
        lines.append(
            _markdown_row(
                [
                    candidate_id,
                    _format_score(mean),
                    _format_score(min_model),
                    _format_score(mean_turns),
                    str(int(prompt_chars)),
                    _format_score(size_ratio),
                    *[_format_score(category_means[cat]) for cat in categories],
                ]
            )
        )
    return lines


def _model_matrix_table(
    scores: list[CandidateModelScore], candidate_ids: list[str], model_ids: list[str]
) -> list[str]:
    """Render per-candidate/per-model mean and turns rows."""
    score_map = {(score.candidate_id, score.model_id): score for score in scores}
    header = ["Candidate", "Model", "Mean", "Turns"]
    lines = [_markdown_row(header), _markdown_separator(len(header))]
    for candidate_id in candidate_ids:
        for model_id in model_ids:
            score = score_map.get((candidate_id, model_id))
            lines.append(
                _markdown_row(
                    [
                        candidate_id,
                        model_id,
                        _format_score(0.0 if score is None else score.mean),
                        _format_score(0.0 if score is None else score.mean_turns),
                    ]
                )
            )
    return lines


def _markdown_row(values: list[str]) -> str:
    """Render one Markdown table row."""
    return "| " + " | ".join(values) + " |"


def _markdown_separator(column_count: int) -> str:
    """Render one Markdown table separator row."""
    return "| " + " | ".join("---" for _ in range(column_count)) + " |"


def _format_score(value: float) -> str:
    """Format scores consistently for reports."""
    return f"{value:.3f}"


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean for report-only aggregations."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _string_field(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"run.json field {key!r} must be a string")
    return value


def _int_field(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise ValueError(f"run.json field {key!r} must be an integer")
    return value


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("run.json field must be a list")
    return [item for item in value if isinstance(item, str)]


def _scores_field(value: object) -> list[CandidateModelScore]:
    if not isinstance(value, list):
        raise ValueError("run.json field 'scores' must be a list")
    scores: list[CandidateModelScore] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("run.json score entries must be objects")
        scores.append(
            CandidateModelScore(
                candidate_id=_string_field(item, "candidate_id"),
                model_id=_string_field(item, "model_id"),
                mean=float(item.get("mean", 0.0)),
                mean_turns=float(item.get("mean_turns", 0.0)),
                per_category=_float_map(item.get("per_category")),
                case_scores=_float_map(item.get("case_scores")),
                api_prompt_chars=int(item.get("api_prompt_chars", 0)),
                prompt_chars=int(item.get("prompt_chars", 0)),
            )
        )
    return scores


def _float_map(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("run.json score maps must be objects")
    return {str(key): float(item) for key, item in value.items() if isinstance(item, int | float)}
