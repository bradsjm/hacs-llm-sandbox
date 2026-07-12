"""Command-line interface for the dev-only eval harness.

Output contract: ``stdout`` stays machine-readable (the run directory plus
leaderboard for ``eval``; factual key/value lines for ``optimize`` and
``report``). All human guidance — "what is it doing", configuration echo,
artifact explanations, and next steps — is written to ``stderr`` so piping
``stdout`` keeps working while interactive users still see the full picture.
"""

import argparse
import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
import os
from pathlib import Path
import signal
import sys
import termios
import tty
from types import FrameType, ModuleType
from typing import cast
import warnings

from custom_components.llm_sandbox.llm_api.prompts import resolve_profile
from dotenv import load_dotenv
from pydantic_evals.reporting import EvaluationReport
from rich.console import Console

from llm_sandbox_evals import experiment, html_report, reports
from llm_sandbox_evals.config import EvalConfig, EvalOutputMode, load_config
from llm_sandbox_evals.experiment import MatrixCellMeta, MatrixCellRef
from llm_sandbox_evals.harness import _select_cases
from llm_sandbox_evals.schema import CaseTrace
from llm_sandbox_evals.terminal import MatrixTerminalReporter

type _TermiosSettings = list[int | list[bytes | int]]

_ESCAPE_DELAY_SECONDS = 0.05
_CLI_ERROR_MAX_CHARS = 500
_DEBUG_ENV_VAR = "LLM_SANDBOX_EVALS_DEBUG"


class _EvalCancelled(Exception):
    """Signal an Escape-initiated interactive eval cancellation."""


class _EscapeWatcher:
    """Read Escape in cbreak mode for one interactive eval and restore the terminal."""

    def __init__(self, on_escape: Callable[[], object]) -> None:
        """Store the matrix-cancellation callback without touching terminal state yet."""
        self._on_escape = on_escape
        self._fd: int | None = None
        self._settings: _TermiosSettings | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._escape_handle: asyncio.TimerHandle | None = None
        self.cancelled = False

    def __enter__(self) -> _EscapeWatcher:
        """Enter cbreak mode and watch stdin only for the active interactive eval."""
        self._fd = sys.stdin.fileno()
        self._settings = cast(_TermiosSettings, termios.tcgetattr(self._fd))
        self._loop = asyncio.get_running_loop()
        try:
            tty.setcbreak(self._fd)
            self._loop.add_reader(self._fd, self._read_input)
        except Exception:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._settings)
            raise
        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        """Remove the reader and restore stdin on every completion or cancellation path."""
        self._cancel_pending_escape()
        if self._loop is not None and self._fd is not None:
            self._loop.remove_reader(self._fd)
        if self._settings is not None and self._fd is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._settings)

    def _read_input(self) -> None:
        """Cancel the current matrix task when the user presses Escape."""
        assert self._fd is not None
        try:
            pressed = os.read(self._fd, 1)
        except OSError:
            return
        self._handle_input(pressed)

    def _handle_input(self, pressed: bytes) -> None:
        """Delay lone Escape cancellation so arrow and Alt sequences remain usable."""
        if self._escape_handle is not None:
            # Branch boundary: any following byte makes the pending Escape part of a sequence.
            self._cancel_pending_escape()
            return
        if pressed != b"\x1b":
            return
        assert self._loop is not None
        self._escape_handle = self._loop.call_later(_ESCAPE_DELAY_SECONDS, self._cancel_lone_escape)

    def _cancel_lone_escape(self) -> None:
        """Cancel the active matrix after Escape remained standalone for one short delay."""
        self._escape_handle = None
        # State mutation point: only a standalone Escape requests matrix cancellation.
        self.cancelled = True
        self._on_escape()

    def _cancel_pending_escape(self) -> None:
        """Discard a pending lone-Escape timer during input continuation or cleanup."""
        if self._escape_handle is not None:
            self._escape_handle.cancel()
            self._escape_handle = None


def main(argv: list[str] | None = None) -> int:
    """Run the eval CLI with one user-facing application error boundary."""
    command: str | None = None
    try:
        # Load a .env file (if present) before any model adapter runs. Existing
        # environment variables take precedence, so explicit exports still win.
        load_dotenv()
        parser = _build_parser()
        args = parser.parse_args(argv)
        command = args.command

        # Branch boundary: argparse leaves subcommand unset when invoked without args.
        if command == "eval":
            return _run_eval(args)
        if command == "report":
            return _run_report(args)
        if command == "optimize":
            return _run_optimize(args)

        parser.print_help(sys.stderr)
        return 2
    except KeyboardInterrupt, asyncio.CancelledError:
        # Terminal contexts unwind before this boundary renders the durable interruption line.
        _say("eval interrupted" if command == "eval" else "interrupted")
        return 130
    except Exception as err:
        # Branch boundary: developers may opt into the original exception for local debugging.
        if os.getenv(_DEBUG_ENV_VAR) == "1":
            raise
        _say(_format_unexpected_error(err))
        return 1


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with newcomer-friendly help."""
    parser = argparse.ArgumentParser(
        prog="llm_sandbox_evals",
        description=(
            "Development-only eval harness for the llm_sandbox Home Assistant integration.\n"
            "It runs the integration's real LLM tools against frozen fixtures, evaluates\n"
            "structured claims and evidence plus action effects, and ranks candidates by binary correctness."
        ),
        epilog=(
            "Examples:\n"
            "  # Offline, no API key — evaluates structured claims and action effects:\n"
            "  python -m llm_sandbox_evals eval --models stub\n"
            "\n"
            "  # Score real models (keys read from env/.env):\n"
            "  python -m llm_sandbox_evals eval --models openai:gpt-4o-mini,anthropic:claude-haiku-4-5,stub\n"
            "\n"
            "  # Re-render a saved run's leaderboard (no model calls):\n"
            "  python -m llm_sandbox_evals report 20260630-164326-318981\n"
            "\n"
            "  # Optimize the prompt with DSPy COPRO (costs real model calls):\n"
            "  python -m llm_sandbox_evals optimize --target-model openrouter/openai/gpt-4o-mini --breadth 5 --depth 2\n"
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
            "frozen Home Assistant snapshot, then evaluates structured claims, grounded evidence, and actions.\n"
            "Results are binary correct/incorrect; provider failures are incomplete. Artifacts are written under the\n"
            "runs directory; a native pydantic-evals summary is printed to stdout."
        ),
        epilog=(
            "Notes:\n"
            "  - --cases accepts case ids OR category names: state, registry, history,\n"
            "    statistics, logbook, automation, action, safety, system.\n"
            "  - --candidates accepts `baseline` and `optimized:<path>` (a saved\n"
            "    optimized_candidate.json from `optimize`).\n"
            "  - --prompt-profile selects one production base prompt profile for\n"
            "    the whole run; it is separate from --candidates.\n"
            "  - A failing provider call (bad key, network) is incomplete and excluded\n"
            "    from the correctness denominator; it never aborts the run.\n"
            "  - Press Escape to cancel an interactive eval run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    eval_parser.add_argument(
        "--models",
        metavar="ID,ID,...",
        help=(
            "comma-separated model ids to evaluate (default: stub). Any Pydantic AI model id works "
            "(e.g. openai:gpt-4o-mini, anthropic:claude-haiku-4-5, "
            "openrouter:anthropic/claude-sonnet-4.6); API keys come from the environment. "
            "'stub' is offline and keyless."
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
        help="production prompt profile id to use for the baseline candidate and runtime settings (default: balanced).",
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
        help="maximum concurrent candidate x model x case matrix cells (default: 5).",
    )
    eval_parser.add_argument(
        "--max-tool-calls",
        type=int,
        metavar="N",
        help="max tool calls per case before recording a limit failure (default: 8).",
    )
    eval_parser.add_argument(
        "--model-timeout",
        type=float,
        metavar="SECONDS",
        help="seconds to wait for one model generation before recording an incomplete provider result (default: 75).",
    )
    eval_parser.add_argument(
        "--reasoning",
        metavar="LEVEL",
        help="reasoning effort forwarded to real models via Pydantic AI provider settings "
        "(OpenRouter/OpenAI reasoning effort; ignored by 'stub' and providers without a reasoning setting).",
    )
    eval_parser.add_argument(
        "--temperature",
        type=float,
        metavar="FLOAT",
        help="sampling temperature forwarded to real models via Pydantic AI provider settings "
        "(default: unset; left to the provider so reasoning-capable models do not warn about "
        "unsupported sampling parameters). Ignored by 'stub'.",
    )
    eval_parser.add_argument(
        "--output-mode",
        choices=("tool", "json-schema"),
        metavar="MODE",
        help="structured eval result protocol: tool (default) or json-schema (provider-native schema output).",
    )


def _add_report_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the `report` subcommand parser."""
    report_parser = subparsers.add_parser(
        "report",
        help="re-render a saved native report",
        description=(
            "Reload a saved report.json and render its analyses without re-running any "
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
    report_parser.add_argument(
        "--html",
        action="store_true",
        help="regenerate report.html from the saved report.json (no model calls)",
    )


def _add_optimize_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the `optimize` subcommand parser."""
    optimize_parser = subparsers.add_parser(
        "optimize",
        help="optimize the API prompt with DSPy COPRO (costs model calls)",
        description=(
            "Use DSPy's COPRO instruction optimizer to rewrite the execute_home_code\n"
            "instruction. The REAL eval harness is the metric: each proposed instruction is\n"
            "evaluated through structured EvalAnswer claims, grounded tool evidence, and action ledgers against the\n"
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
        help="model id to optimize against (required; must be a real model, not 'stub'; optimize uses DSPy, which accepts litellm-style ids such as openrouter/...).",
    )
    optimize_parser.add_argument(
        "--proposer-model",
        metavar="ID",
        help="model id COPRO uses to propose prompt rewrites (default: the target model; DSPy accepts litellm-style ids such as openrouter/...).",
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
            "change the reported baseline_correct_rate/optimized_correct_rate (those stay raw quality)."
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
        help="production prompt profile id to use for the baseline candidate and runtime settings (default: balanced).",
    )
    optimize_parser.add_argument(
        "--reasoning",
        metavar="LEVEL",
        help="reasoning effort forwarded to cross-eval harness models via Pydantic AI provider settings.",
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
        runs_dir=Path(args.runs_dir) if args.runs_dir else base_config.runs_dir,
        output_mode=cast(EvalOutputMode, args.output_mode or base_config.output_mode),
        concurrency=args.concurrency if args.concurrency else base_config.concurrency,
        max_tool_calls=args.max_tool_calls if args.max_tool_calls else base_config.max_tool_calls,
        model_timeout=args.model_timeout if args.model_timeout else base_config.model_timeout,
        reasoning_effort=args.reasoning,
        temperature=args.temperature,
    )
    run_id = _derive_run_id()
    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    try:
        with MatrixTerminalReporter() as reporter:
            report = asyncio.run(_run_eval_matrix(config, run_id, reporter))
    except _EvalCancelled:
        sys.stderr.write("eval cancelled\n")
        return 130
    else:
        run_dir = reports.write_report_json(report, config, run_id=run_id)
        report_html = html_report.write_html(run_dir)
        reporter.finish(
            overall_correct_rate=experiment.overall_correct_rate(report),
            run_dir=str(run_dir),
            report_html=str(report_html),
        )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)

    # stdout stays machine-readable: the run directory and compact native analysis facts.
    sys.stdout.write(f"run_dir: {run_dir}\n")
    sys.stdout.write(f"report_html: {report_html}\n")
    sys.stdout.write("\n".join(experiment.matrix_summary_lines(report)) + "\n")
    return 0


async def _run_eval_matrix(
    config: EvalConfig, run_id: str, reporter: MatrixTerminalReporter
) -> EvaluationReport[MatrixCellRef, CaseTrace, MatrixCellMeta]:
    """Run one matrix, allowing Escape to cancel only interactive terminal sessions."""
    # Branch boundary: redirected streams retain their existing non-interactive behavior.
    if not _is_interactive_eval():
        return await experiment.run_matrix(config, run_id=run_id, on_event=reporter.handle)
    current_task = asyncio.current_task()
    assert current_task is not None
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, current_task.cancel)
    try:
        with _EscapeWatcher(current_task.cancel) as watcher:
            try:
                return await experiment.run_matrix(config, run_id=run_id, on_event=reporter.handle)
            except asyncio.CancelledError:
                if watcher.cancelled:
                    raise _EvalCancelled from None
                raise
    finally:
        loop.remove_signal_handler(signal.SIGINT)


def _is_interactive_eval() -> bool:
    """Return whether both streams support the cbreak Escape interaction."""
    return sys.stdin.isatty() and sys.stderr.isatty()


def _run_report(args: argparse.Namespace) -> int:
    """Load a saved run.json and render its leaderboard."""
    base_config = load_config()
    runs_dir = Path(args.runs_dir) if args.runs_dir else base_config.runs_dir
    run_id = args.run_id_option or args.run_id
    # Branch boundary: report needs exactly one run id source.
    if run_id is None:
        sys.stderr.write("error: report requires a run_id\n")
        return 2

    report_json = runs_dir / run_id / "report.json"
    if not report_json.exists():
        sys.stderr.write(f"error: run not found: {report_json}\n")
        return 1

    report = reports.load_report(report_json.parent)
    if args.html:
        report_html = html_report.write_html(report_json.parent)
        sys.stdout.write(f"report_html: {report_html}\n")
        return 0

    _say("(llm sandbox evals) re-rendering the native report summary from report.json (no model calls).\n")
    report.print(
        console=Console(stderr=True),
        include_output=False,
        include_expected_output=False,
        include_durations=True,
    )
    sys.stdout.write("\n".join(experiment.matrix_summary_lines(report)) + "\n")
    return 0


def _run_optimize(args: argparse.Namespace) -> int:
    """Run DSPy optimization and print the exported candidate summary."""
    optimize_dspy = _load_optimizer()

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
        runs_dir=Path(args.runs_dir) if args.runs_dir else base_config.runs_dir,
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
            baseline_correct_rate=result.baseline_correct_rate,
            optimized_correct_rate=result.optimized_correct_rate,
            optimized_prompt_chars=result.optimized_prompt_chars,
            baseline_prompt_chars=result.baseline_prompt_chars,
            size_ratio=result.size_ratio,
            optimized_full_correct_rate=result.optimized_full_correct_rate,
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
        f"baseline_correct_rate: {result.baseline_correct_rate:.3f}",
        f"optimized_correct_rate: {result.optimized_correct_rate:.3f}",
        f"baseline_prompt_chars: {result.baseline_prompt_chars}",
        f"optimized_prompt_chars: {result.optimized_prompt_chars}",
        f"size_ratio: {result.size_ratio:.3f}",
        f"optimized_full_correct_rate: {result.optimized_full_correct_rate:.3f}",
        f"optimized_candidate: {result.candidate_path}",
        f"optimized_prompt: {result.candidate_path.parent / 'optimized_prompt.md'}",
    ]
    # Branch boundary: cross-eval artifacts exist only when explicitly requested.
    if result.cross_eval_run_dir is not None:
        lines.append(f"cross_eval_run_dir: {result.cross_eval_run_dir}")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _raise_keyboard_interrupt(_signal_number: int, _frame: FrameType | None) -> None:
    """Interrupt the terminal setup window before asyncio installs its handler."""
    raise KeyboardInterrupt


def _load_optimizer() -> ModuleType:
    """Import the optional optimizer while containing one known DSPy warning."""
    with warnings.catch_warnings():
        # Boundary constraint: contain only DSPy's known import-time field-prefix deprecation.
        warnings.filterwarnings(
            "ignore",
            message=r".*'prefix' argument in InputField/OutputField is deprecated.*",
            category=DeprecationWarning,
            module=r"^dspy\.",
        )
        from llm_sandbox_evals import optimize_dspy

    return optimize_dspy


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
        "as its metric: every proposed instruction is evaluated through structured answers,\n"
        "grounded tool evidence, and action ledgers against the target model. The best-correctness\n"
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
    baseline_correct_rate: float,
    optimized_correct_rate: float,
    optimized_prompt_chars: int,
    baseline_prompt_chars: int,
    size_ratio: float,
    optimized_full_correct_rate: float,
    candidate_path: Path,
    cross_eval_run_dir: Path | None,
    prompt_profile: str,
) -> str:
    """Build the post-run interpretation + next-steps footer for `optimize`."""
    prompt_path = candidate_path.parent / "optimized_prompt.md"
    delta = optimized_correct_rate - baseline_correct_rate
    delta_str = f"{'+' if delta >= 0 else ''}{delta:.3f}"
    cross_eval_line = ""
    # Branch boundary: cross-eval is optional and may not have run.
    if cross_eval_run_dir is not None:
        cross_eval_line = (
            f"\n  cross-eval report: {cross_eval_run_dir / 'report.json'}\n"
            "    (baseline vs optimized across the requested model matrix)"
        )
    return (
        "\nOptimization complete.\n\n"
        f"  baseline_correct_rate   : {baseline_correct_rate:.3f}   (production profile {prompt_profile!r}, on the target model)\n"
        f"  optimized_correct_rate  : {optimized_correct_rate:.3f}   (best COPRO rewrite)   delta {delta_str}\n"
        f"  optimized_chars : {optimized_prompt_chars}   (baseline {baseline_prompt_chars}; ratio {size_ratio:.3f})\n"
        f"  optimized_full_correct_rate : {optimized_full_correct_rate:.3f}   (full case suite, not just the trainset)\n"
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


def _format_unexpected_error(error: Exception) -> str:
    """Return one bounded diagnostic line including the useful exception cause chain."""
    parts: list[str] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        detail = " ".join(str(current).split())
        parts.append(f"{type(current).__name__}: {detail}" if detail else type(current).__name__)
        current = current.__cause__ or current.__context__
    message = "error: " + " (caused by ".join(parts) + ")" * (len(parts) - 1)
    return message if len(message) <= _CLI_ERROR_MAX_CHARS else f"{message[: _CLI_ERROR_MAX_CHARS - 3]}..."


def _derive_run_id() -> str:
    """Return a filesystem-friendly run id."""
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")


def _csv_arg(value: str | None) -> list[str] | None:
    """Parse a comma-separated CLI value into a list, preserving item order."""
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]
