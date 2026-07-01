"""Command-line interface for the dev-only eval harness."""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from llm_sandbox_evals.config import EvalConfig, load_config
from llm_sandbox_evals.harness import run_matrix
from llm_sandbox_evals.reports import load_run_json, render_leaderboard_from_scores, write_run


def main(argv: list[str] | None = None) -> int:
    """Run the eval CLI."""
    # Load a .env file (if present) before any model adapter runs. Existing
    # environment variables take precedence, so explicit exports still win.
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Branch boundary: argparse leaves subcommand unset when invoked without args.
    if args.command == "eval":
        return _run_eval(args)
    if args.command == "report":
        return _run_report(args)
    if args.command == "optimize":
        return _run_optimize(args)

    parser.print_help(sys.stderr)
    return 2


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser."""
    parser = argparse.ArgumentParser(prog="llm_sandbox_evals")
    subparsers = parser.add_subparsers(dest="command")

    eval_parser = subparsers.add_parser("eval", help="run the eval matrix")
    eval_parser.add_argument("--models", help="comma-separated model ids")
    eval_parser.add_argument("--candidates", help="comma-separated prompt candidate ids")
    eval_parser.add_argument("--cases", help="comma-separated case ids or categories")
    eval_parser.add_argument("--runs-dir", type=Path, help="directory for run artifacts")
    eval_parser.add_argument(
        "--concurrency", type=int, help="max concurrent model calls per candidate/model (default 5)"
    )
    eval_parser.add_argument(
        "--reasoning",
        help="litellm reasoning effort level passed to real models (e.g. minimal/low/medium/high)",
    )

    report_parser = subparsers.add_parser("report", help="render a saved run leaderboard")
    report_parser.add_argument("run_id", nargs="?", help="run id under the runs directory")
    report_parser.add_argument("--run-id", dest="run_id_option", help="run id under the runs directory")
    report_parser.add_argument("--runs-dir", type=Path, help="directory containing run artifacts")

    optimize_parser = subparsers.add_parser("optimize", help="optimize the API prompt with DSPy COPRO")
    optimize_parser.add_argument("--target-model", help="model id to optimize against")
    optimize_parser.add_argument("--proposer-model", help="model id used to propose prompt rewrites")
    optimize_parser.add_argument("--breadth", type=int, help="COPRO breadth (default 5)")
    optimize_parser.add_argument("--depth", type=int, help="COPRO depth (default 2)")
    optimize_parser.add_argument("--cases", help="comma-separated case ids or categories")
    optimize_parser.add_argument(
        "--reasoning",
        help="reasoning effort level passed to cross-eval harness models (e.g. minimal/low/medium/high)",
    )
    optimize_parser.add_argument("--cross-eval-models", help="comma-separated model ids for baseline vs optimized eval")
    optimize_parser.add_argument("--runs-dir", type=Path, help="directory for run artifacts")
    return parser


def _run_eval(args: argparse.Namespace) -> int:
    """Run the matrix and write artifacts."""
    base_config = load_config()
    config = EvalConfig(
        models=_csv_arg(args.models) or base_config.models,
        candidates=_csv_arg(args.candidates) or base_config.candidates,
        cases=_csv_arg(args.cases) if args.cases is not None else base_config.cases,
        homes=base_config.homes,
        runs_dir=args.runs_dir or base_config.runs_dir,
        concurrency=args.concurrency if args.concurrency else base_config.concurrency,
        reasoning_effort=args.reasoning,
    )
    result = asyncio.run(run_matrix(config))
    run_dir = write_run(result, config.runs_dir)
    sys.stdout.write(f"{run_dir}\n\n")
    sys.stdout.write((run_dir / "leaderboard.md").read_text(encoding="utf-8"))
    return 0


def _run_report(args: argparse.Namespace) -> int:
    """Load a saved run.json and render its leaderboard."""
    base_config = load_config()
    runs_dir = args.runs_dir or base_config.runs_dir
    run_id = args.run_id_option or args.run_id
    # Branch boundary: report needs exactly one run id source.
    if run_id is None:
        sys.stderr.write("error: report requires a run_id\n")
        return 2

    run_json = runs_dir / run_id / "run.json"
    if not run_json.exists():
        sys.stderr.write(f"error: run not found: {run_json}\n")
        return 1

    loaded_run_id, created_at, case_count, candidate_ids, model_ids, scores = load_run_json(run_json)
    sys.stdout.write(
        render_leaderboard_from_scores(
            scores=scores,
            run_id=loaded_run_id,
            created_at=created_at,
            case_count=case_count,
            candidate_ids=candidate_ids,
            model_ids=model_ids,
        )
    )
    return 0


def _run_optimize(args: argparse.Namespace) -> int:
    """Run DSPy optimization and print the exported candidate summary."""
    # Lazy import keeps the offline eval/report paths usable without importing DSPy.
    from llm_sandbox_evals import optimize_dspy

    base_config = load_config()
    config = EvalConfig(
        models=base_config.models,
        candidates=base_config.candidates,
        cases=_csv_arg(args.cases) if args.cases is not None else base_config.cases,
        homes=None,
        runs_dir=args.runs_dir or base_config.runs_dir,
        reasoning_effort=args.reasoning,
        target_model=args.target_model,
        proposer_model=args.proposer_model,
        breadth=args.breadth or 5,
        depth=args.depth or 2,
        cross_eval_models=_csv_arg(args.cross_eval_models),
    )
    try:
        result = optimize_dspy.run_optimize(config)
    except ValueError as err:
        sys.stderr.write(f"error: {err}\n")
        return 2

    run_dir = result.candidate_path.parent
    sys.stdout.write(
        "\n".join(
            (
                f"run_dir: {run_dir}",
                f"target_model: {result.target_model}",
                f"baseline_mean: {result.baseline_mean:.3f}",
                f"optimized_mean: {result.optimized_mean:.3f}",
                f"optimized_candidate: {result.candidate_path}",
            )
        )
        + "\n"
    )
    # Branch boundary: cross-eval artifacts exist only when explicitly requested.
    if result.cross_eval_run_dir is not None:
        sys.stdout.write(f"cross_eval_run_dir: {result.cross_eval_run_dir}\n")
    return 0


def _csv_arg(value: str | None) -> list[str] | None:
    """Parse a comma-separated CLI value into a list, preserving item order."""
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]
