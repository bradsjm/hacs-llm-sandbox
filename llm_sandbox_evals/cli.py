"""Command-line interface for the dev-only eval harness.

Output contract: ``stdout`` stays machine-readable (the run directory plus
leaderboard for ``eval``; factual key/value lines for ``optimize`` and
``report``). All human guidance — "what is it doing", configuration echo,
artifact explanations, and next steps — is written to ``stderr`` so piping
``stdout`` keeps working while interactive users still see the full picture.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from custom_components.llm_sandbox.llm_api.prompts import resolve_profile
from dotenv import load_dotenv

from llm_sandbox_evals.config import EvalConfig, load_config
from llm_sandbox_evals.harness import _select_cases, run_matrix
from llm_sandbox_evals.html_report import write_html
from llm_sandbox_evals.reports import load_results, load_run_json, render_leaderboard_from_scores, write_run
from llm_sandbox_evals.tui import LiveReporter, render_failures, render_leaderboard, stderr_console

_STUB_NOTE = (
    '"stub" is a keyless, deterministic pipeline-checker: great for verifying the\n'
    "harness end to end, but low stub scores are expected and are not a quality\n"
    "signal. Pass real model ids (e.g. gpt-4o-mini) to measure prompt quality."
)


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
    """Build the top-level argparse parser with newcomer-friendly help."""
    parser = argparse.ArgumentParser(
        prog="llm_sandbox_evals",
        description=(
            "Development-only eval harness for the llm_sandbox Home Assistant integration.\n"
            "It runs the integration's real LLM tools against frozen fixtures, scores each\n"
            "operation deterministically, and ranks prompt candidates across a matrix of models."
        ),
        epilog=(
            "Examples:\n"
            "  # Offline, no API key — validates the whole pipeline:\n"
            "  python -m llm_sandbox_evals eval --models stub\n"
            "\n"
            "  # Score real models (keys read from env/.env):\n"
            "  python -m llm_sandbox_evals eval --models gpt-4o-mini,claude-haiku-4-5,stub\n"
            "\n"
            "  # Re-render a saved run's leaderboard (no model calls):\n"
            "  python -m llm_sandbox_evals report 20260630-164326-318981\n"
            "\n"
            "  # Optimize the prompt with DSPy COPRO (costs real model calls):\n"
            "  python -m llm_sandbox_evals optimize --target-model gpt-4o-mini --breadth 5 --depth 2\n"
            "\n"
            "See `python -m llm_sandbox_evals <command> -h` for command-specific flags."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    _add_eval_parser(subparsers)
    _add_report_parser(subparsers)
    _add_optimize_parser(subparsers)
    return parser


def _add_eval_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the `eval` subcommand parser."""
    eval_parser = subparsers.add_parser(
        "eval",
        help="run the candidate x model x case eval matrix",
        description=(
            "Run the eval matrix: for every (prompt candidate x language model x test case),\n"
            "ask the model to use the available tools over one or more turns against a\n"
            "frozen Home Assistant snapshot, and score the result in 0.0-1.0. Artifacts are written under the\n"
            "runs directory; the leaderboard is printed to stdout."
        ),
        epilog=(
            "Notes:\n"
            "  - --cases accepts case ids OR category names: state_read, registry_read,\n"
            "    recorder_read, action_allowed, action_blocked, complex.\n"
            "  - --candidates accepts `baseline` and `optimized:<path>` (a saved\n"
            "    optimized_candidate.json from `optimize`).\n"
            "  - --prompt-profile selects one production base prompt profile for\n"
            "    the whole run; it is separate from --candidates.\n"
            "  - A failing model call (bad key, network) scores 0.0 for that cell and\n"
            "    never aborts the run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    eval_parser.add_argument(
        "--models",
        metavar="ID,ID,...",
        help=(
            "comma-separated model ids to evaluate (default: stub). Any LiteLLM id works "
            "(e.g. gpt-4o-mini, claude-haiku-4-5, openrouter/...); API keys come from the "
            "environment. 'stub' is offline and keyless."
        ),
    )
    eval_parser.add_argument(
        "--candidates",
        metavar="ID,ID,...",
        help="comma-separated prompt candidate ids (default: baseline). Use optimized:<path> "
        "to load a saved optimized_candidate.json.",
    )
    eval_parser.add_argument(
        "--prompt-profile",
        metavar="PROFILE_ID",
        help="production prompt profile id to use for the baseline candidate and runtime settings (default: standard).",
    )
    eval_parser.add_argument(
        "--cases",
        metavar="ID|CATEGORY,...",
        help="comma-separated case ids or category names (default: all cases).",
    )
    eval_parser.add_argument(
        "--runs-dir",
        metavar="PATH",
        help="directory for run artifacts (default: eval_data/runs).",
    )
    eval_parser.add_argument(
        "--concurrency",
        type=int,
        metavar="N",
        help="max concurrent model calls per candidate/model (default: 5).",
    )
    eval_parser.add_argument(
        "--max-turns",
        type=int,
        metavar="N",
        help="max tool-calling turns per case before forcing a final answer (default: 5).",
    )
    eval_parser.add_argument(
        "--model-timeout",
        type=float,
        metavar="SECONDS",
        help="seconds to wait for one model generation before recording model_error (default: 75).",
    )
    eval_parser.add_argument(
        "--reasoning",
        metavar="LEVEL",
        help="reasoning effort forwarded to real models via LiteLLM "
        "(e.g. minimal/low/medium/high, or none to disable a reasoning model). "
        "Ignored by 'stub'.",
    )


def _add_report_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the `report` subcommand parser."""
    report_parser = subparsers.add_parser(
        "report",
        help="re-render a saved run's leaderboard",
        description=(
            "Reload a saved run.json and render its leaderboard without re-running any "
            "model. No API keys are used and no model calls are made."
        ),
    )
    report_parser.add_argument("run_id", nargs="?", metavar="RUN_ID", help="run id under the runs directory")
    report_parser.add_argument(
        "--run-id",
        dest="run_id_option",
        metavar="RUN_ID",
        help="run id under the runs directory (alternative to the positional argument).",
    )
    report_parser.add_argument(
        "--runs-dir",
        metavar="PATH",
        help="directory containing run artifacts (default: eval_data/runs).",
    )


def _add_optimize_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the `optimize` subcommand parser."""
    optimize_parser = subparsers.add_parser(
        "optimize",
        help="optimize the API prompt with DSPy COPRO (costs model calls)",
        description=(
            "Use DSPy's COPRO instruction optimizer to rewrite the execute_home_code\n"
            "instruction. The REAL eval harness is the metric: each proposed instruction is\n"
            "scored by the existing pipeline (parse -> run tool -> check -> score) against the\n"
            "target model. The winning instruction is exported for human review; production\n"
            "prompts.py is never auto-patched."
        ),
        epilog=(
            "Cost: optimizer calls scale roughly as breadth x depth x trainset cases, plus a\n"
            "baseline and an optimized eval, plus the optional cross-eval. Keep --breadth,\n"
            "--depth, and --cases small to bound spend.\n"
            "\n"
            "Reasoning levels: minimal/low/medium/high, or none to disable a reasoning model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    optimize_parser.add_argument(
        "--target-model",
        required=True,
        metavar="ID",
        help="model id to optimize against (required; must be a real model, not 'stub').",
    )
    optimize_parser.add_argument(
        "--proposer-model",
        metavar="ID",
        help="model id COPRO uses to propose prompt rewrites (default: the target model).",
    )
    optimize_parser.add_argument(
        "--target-reasoning",
        metavar="LEVEL",
        help="reasoning effort for the target model during DSPy scoring and baseline/optimized eval.",
    )
    optimize_parser.add_argument(
        "--proposer-reasoning",
        metavar="LEVEL",
        help="reasoning effort for the proposer model during DSPy.",
    )
    optimize_parser.add_argument(
        "--breadth",
        type=int,
        metavar="N",
        help="COPRO search breadth (default: 5). Cost scales with this.",
    )
    optimize_parser.add_argument(
        "--depth",
        type=int,
        metavar="N",
        help="COPRO search depth (default: 2). Cost scales with this.",
    )
    optimize_parser.add_argument(
        "--length-penalty",
        type=float,
        metavar="COEFF",
        help=(
            "penalty coefficient applied to api_prompt growth during COPRO candidate "
            "selection (default: 0.02). Higher values more aggressively tie-break toward "
            "smaller prompts at equal quality; 0 disables size-aware selection. Does not "
            "change the reported baseline_mean/optimized_mean (those stay raw quality)."
        ),
    )
    optimize_parser.add_argument(
        "--cases",
        metavar="ID|CATEGORY,...",
        help="case ids or categories used as the optimization trainset (default: all cases). "
        "Keep small to bound cost.",
    )
    optimize_parser.add_argument(
        "--prompt-profile",
        metavar="PROFILE_ID",
        help="production prompt profile id to use for the baseline candidate and runtime settings (default: standard).",
    )
    optimize_parser.add_argument(
        "--reasoning",
        metavar="LEVEL",
        help="reasoning effort forwarded to cross-eval harness models.",
    )
    optimize_parser.add_argument(
        "--cross-eval-models",
        metavar="ID,ID,...",
        help="comma-separated model ids for a baseline-vs-optimized leaderboard (default: off; no cross-eval is run).",
    )
    optimize_parser.add_argument(
        "--runs-dir",
        metavar="PATH",
        help="directory for run artifacts (default: eval_data/runs).",
    )


def _run_eval(args: argparse.Namespace) -> int:
    """Run the matrix and write artifacts."""
    base_config = load_config()
    prompt_profile = args.prompt_profile or base_config.prompt_profile
    try:
        resolve_profile(prompt_profile)
    except ValueError as err:
        sys.stderr.write(f"error: {err}\n")
        return 2
    config = EvalConfig(
        models=_csv_arg(args.models) or base_config.models,
        candidates=_csv_arg(args.candidates) or base_config.candidates,
        prompt_profile=prompt_profile,
        cases=_csv_arg(args.cases) if args.cases is not None else base_config.cases,
        homes=base_config.homes,
        runs_dir=args.runs_dir or base_config.runs_dir,
        concurrency=args.concurrency if args.concurrency else base_config.concurrency,
        max_turns=args.max_turns if args.max_turns else base_config.max_turns,
        model_timeout=args.model_timeout if args.model_timeout else base_config.model_timeout,
        efficiency_k=base_config.efficiency_k,
        efficiency_floor=base_config.efficiency_floor,
        reasoning_effort=args.reasoning,
    )
    selected_cases = _select_cases(config.cases, config.homes)
    _say(_eval_banner(config, len(selected_cases)))

    console = stderr_console()
    with LiveReporter(console) as reporter:
        result = asyncio.run(run_matrix(config, reporter=reporter))
    run_dir = write_run(result, config.runs_dir)
    _say(_eval_footer(run_dir))
    render_leaderboard(
        console,
        scores=result.scores,
        run_id=result.run_id,
        created_at=result.created_at,
        case_count=len(result.case_ids),
        candidate_ids=result.candidate_ids,
        model_ids=result.model_ids,
    )
    render_failures(console, load_results(run_dir / "results.jsonl"))

    # stdout stays machine-readable: the run directory then the leaderboard.
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

    _say("(llm sandbox evals) re-rendering the leaderboard from saved scores (no model calls).\n")
    loaded_run_id, created_at, case_count, candidate_ids, model_ids, scores = load_run_json(run_json)
    html_path = write_html(run_json.parent)
    console = stderr_console()
    render_leaderboard(
        console,
        scores=scores,
        run_id=loaded_run_id,
        created_at=created_at,
        case_count=case_count,
        candidate_ids=candidate_ids,
        model_ids=model_ids,
    )
    render_failures(console, load_results(run_json.parent / "results.jsonl"))
    _say(f"HTML report: {html_path}\n")
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
    prompt_profile = args.prompt_profile or base_config.prompt_profile
    try:
        resolve_profile(prompt_profile)
    except ValueError as err:
        sys.stderr.write(f"error: {err}\n")
        return 2
    config = EvalConfig(
        models=base_config.models,
        candidates=base_config.candidates,
        prompt_profile=prompt_profile,
        cases=_csv_arg(args.cases) if args.cases is not None else base_config.cases,
        homes=None,
        runs_dir=args.runs_dir or base_config.runs_dir,
        reasoning_effort=args.reasoning,
        target_model=args.target_model,
        proposer_model=args.proposer_model,
        target_reasoning_effort=args.target_reasoning,
        proposer_reasoning_effort=args.proposer_reasoning,
        breadth=args.breadth or 5,
        depth=args.depth or 2,
        length_penalty=args.length_penalty if args.length_penalty is not None else base_config.length_penalty,
        cross_eval_models=_csv_arg(args.cross_eval_models),
    )
    _say(_optimize_banner(config))
    try:
        result = optimize_dspy.run_optimize(config)
    except ValueError as err:
        sys.stderr.write(f"error: {err}\n")
        return 2

    _say(
        _optimize_footer(
            baseline_mean=result.baseline_mean,
            optimized_mean=result.optimized_mean,
            optimized_prompt_chars=result.optimized_prompt_chars,
            baseline_prompt_chars=result.baseline_prompt_chars,
            size_ratio=result.size_ratio,
            optimized_full_mean=result.optimized_full_mean,
            candidate_path=result.candidate_path,
            cross_eval_run_dir=result.cross_eval_run_dir,
            prompt_profile=config.prompt_profile,
        )
    )

    # stdout stays machine-readable: factual result keys only.
    run_dir = result.candidate_path.parent
    lines = [
        f"run_dir: {run_dir}",
        f"target_model: {result.target_model}",
        f"proposer_model: {result.proposer_model}",
        f"baseline_mean: {result.baseline_mean:.3f}",
        f"optimized_mean: {result.optimized_mean:.3f}",
        f"baseline_prompt_chars: {result.baseline_prompt_chars}",
        f"optimized_prompt_chars: {result.optimized_prompt_chars}",
        f"size_ratio: {result.size_ratio:.3f}",
        f"optimized_full_mean: {result.optimized_full_mean:.3f}",
        f"optimized_candidate: {result.candidate_path}",
        f"optimized_prompt: {result.candidate_path.parent / 'optimized_prompt.md'}",
    ]
    # Branch boundary: cross-eval artifacts exist only when explicitly requested.
    if result.cross_eval_run_dir is not None:
        lines.append(f"cross_eval_run_dir: {result.cross_eval_run_dir}")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _eval_banner(config: EvalConfig, case_count: int) -> str:
    """Build the pre-run orientation banner for `eval`."""
    cases_field = f"all ({case_count})" if config.cases is None else f"{', '.join(config.cases)} ({case_count})"
    reasoning = config.reasoning_effort or "(none)"
    return (
        "llm_sandbox evals - running the eval matrix\n\n"
        "For every (prompt candidate x language model x test case), this harness:\n"
        "  1. renders native tool-calling messages and function schemas,\n"
        "  2. lets the model use available tools over one or more bounded turns, and\n"
        "  3. scores the final outcome plus turn efficiency in 0.0-1.0.\n\n"
        "Candidates rank by mean score; ties break toward the candidate that holds up\n"
        'best on the worst model (the "MinModel" column). A higher mean is better.\n\n'
        "Config:\n"
        f"  models      : {', '.join(config.models)}\n"
        f"  candidates  : {', '.join(config.candidates)}\n"
        f"  prompt profile: {config.prompt_profile}\n"
        f"  cases       : {cases_field}\n"
        f"  runs dir    : {config.runs_dir}\n"
        f"  concurrency : {config.concurrency}\n"
        f"  max turns   : {config.max_turns}\n"
        f"  model timeout: {config.model_timeout:g}s\n"
        f"  reasoning   : {reasoning}\n\n"
        f"{_STUB_NOTE}\n"
    )


def _eval_footer(run_dir: Path) -> str:
    """Build the post-run artifacts/next-steps footer for `eval`."""
    return (
        f"\nWrote artifacts to {run_dir}:\n"
        "  run.json          run metadata + per-(candidate,model) scores (no API keys)\n"
        "  leaderboard.md    the table printed below\n"
        "  report.html       self-contained browser report for demos\n"
        "  results.jsonl     one row per (case, candidate, model) with per-check outcomes\n"
        "  failures.jsonl    subset scoring 0.0 or failing a required gate\n"
        "  traces/*.json     full prompt, model output, tool result, actions per cell\n\n"
        "Re-render just the leaderboard later (no model calls):\n"
        f"  python -m llm_sandbox_evals report {run_dir.name}\n\n"
        "Leaderboard (ranked by mean; MinModel = worst-model robustness):\n"
    )


def _optimize_banner(config: EvalConfig) -> str:
    """Build the pre-run orientation + cost banner for `optimize`."""
    trainset_count = len(_select_cases(config.cases, config.homes))
    cross_eval = ", ".join(config.cross_eval_models) if config.cross_eval_models else "off"
    target_effort = config.target_reasoning_effort or "(default)"
    proposer_effort = config.proposer_reasoning_effort or "(default)"
    cross_effort = config.reasoning_effort or "(default)"
    return (
        "llm_sandbox evals - DSPy COPRO prompt optimization\n\n"
        "COPRO rewrites the execute_home_code instruction and uses the REAL eval harness\n"
        "as its metric: every proposed instruction is scored by the existing pipeline\n"
        "(parse -> run tool -> check -> score) against the target model. The best-scoring\n"
        "instruction is exported for human review. Production prompts.py is never changed.\n\n"
        "COST: this calls real, paid models. Optimizer calls scale roughly as\n"
        "breadth x depth x trainset, plus a baseline and an optimized eval (and the\n"
        "optional cross-eval). DSPy will print its own progress below.\n\n"
        "Config:\n"
        f"  target model    : {config.target_model}   (optimized against; must be real)\n"
        f"  proposer model  : {config.proposer_model or '(same as target)'}   (proposes rewrites)\n"
        f"  prompt profile  : {config.prompt_profile}\n"
        f"  breadth / depth : {config.breadth} / {config.depth}   (COPRO search shape)\n"
        f"  trainset        : {trainset_count} case(s)   (--cases to shrink cost)\n"
        f"  cross-eval      : {cross_eval}\n"
        f"  reasoning       : target={target_effort} proposer={proposer_effort} cross-eval={cross_effort}\n\n"
        "Optimizing... (this can take a while; DSPy scores many candidate instructions.)\n"
    )


def _optimize_footer(
    *,
    baseline_mean: float,
    optimized_mean: float,
    optimized_prompt_chars: int,
    baseline_prompt_chars: int,
    size_ratio: float,
    optimized_full_mean: float,
    candidate_path: Path,
    cross_eval_run_dir: Path | None,
    prompt_profile: str,
) -> str:
    """Build the post-run interpretation + next-steps footer for `optimize`."""
    prompt_path = candidate_path.parent / "optimized_prompt.md"
    delta = optimized_mean - baseline_mean
    delta_str = f"{'+' if delta >= 0 else ''}{delta:.3f}"
    cross_eval_line = ""
    # Branch boundary: cross-eval is optional and may not have run.
    if cross_eval_run_dir is not None:
        cross_eval_line = (
            f"\n  cross-eval leaderboard: {cross_eval_run_dir / 'leaderboard.md'}\n"
            "    (baseline vs optimized across the requested model matrix)"
        )
    return (
        "\nOptimization complete.\n\n"
        f"  baseline_mean   : {baseline_mean:.3f}   (production profile {prompt_profile!r}, on the target model)\n"
        f"  optimized_mean  : {optimized_mean:.3f}   (best COPRO rewrite)   delta {delta_str}\n"
        f"  optimized_chars : {optimized_prompt_chars}   (baseline {baseline_prompt_chars}; ratio {size_ratio:.3f})\n"
        f"  optimized_full_mean : {optimized_full_mean:.3f}   (mean on the FULL case suite, not just the trainset)\n"
        f"  optimized prompt: {prompt_path}   (the ONLY text COPRO changed - review this)\n"
        f"{cross_eval_line}\n\n"
        "Next steps:\n"
        "  - Review optimized_prompt.md before trusting it.\n"
        "  - Compare baseline vs optimized across more models:\n"
        "      python -m llm_sandbox_evals eval \\\n"
        f"        --prompt-profile {prompt_profile} \\\n"
        f"        --candidates baseline,optimized:{candidate_path} \\\n"
        "        --models <model-a>,<model-b>,stub\n"
        "  - To ship it, copy the approved text into\n"
        "    custom_components/llm_sandbox/llm_api/prompts/profiles.py by hand.\n"
    )


def _say(text: str) -> None:
    """Write an explanatory block to stderr, keeping stdout machine-readable."""
    if text:
        sys.stderr.write(text if text.endswith("\n") else text + "\n")
    sys.stderr.flush()


def _csv_arg(value: str | None) -> list[str] | None:
    """Parse a comma-separated CLI value into a list, preserving item order."""
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]
