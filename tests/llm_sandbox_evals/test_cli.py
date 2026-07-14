import asyncio
from collections import Counter
from collections.abc import Callable
from contextlib import nullcontext
import json
import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import sys
import termios
import time
from types import SimpleNamespace
from typing import Never, cast
import warnings

from llm_sandbox_evals.cli import main
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.experiment import (
    LanePhaseCallback,
    LanePhaseEvent,
    MatrixCellMeta,
    MatrixCellRef,
    MatrixEventCallback,
)
from llm_sandbox_evals.schema import CaseTrace, RunDescriptor
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals.reporting import EvaluationReport
import pytest

from llm_sandbox_evals import agent_runner, cli, logfire_config, reports


def test_optimizer_loader_contains_known_dspy_warning_only() -> None:
    optimizer = cli._load_optimizer()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        warnings.warn("unrelated project warning", DeprecationWarning, stacklevel=1)

    assert optimizer.__name__ == "llm_sandbox_evals.optimize_dspy"
    assert [str(item.message) for item in captured] == ["unrelated project warning"]


async def _raise_keyboard_interrupt(*_args: object) -> object:
    """Raise a process-style interruption from the running eval coroutine."""
    raise KeyboardInterrupt


async def _raise_cancelled_error(*_args: object) -> object:
    """Raise an escaping cancellation from the running eval coroutine."""
    raise asyncio.CancelledError


async def _raise_runtime_error(*_args: object) -> object:
    """Raise a deterministic operational failure from the running eval coroutine."""
    raise RuntimeError("matrix runtime failed")


def _raise_artifact_error(*_args: object, **_kwargs: object) -> object:
    """Raise a deterministic report artifact failure."""
    raise RuntimeError("artifact write failed")


def _raise_keyboard_interrupt_sync(*_args: object, **_kwargs: object) -> object:
    """Raise a synchronous interruption from the post-evaluation persistence path."""
    raise KeyboardInterrupt


def _raise_report_error(*_args: object, **_kwargs: object) -> object:
    """Raise a deterministic report loading failure."""
    raise RuntimeError("report load failed")


def _raise_optimizer_error(*_args: object, **_kwargs: object) -> object:
    """Raise a deterministic optimizer runtime failure."""
    raise RuntimeError("optimizer runtime failed")


def _failing_optimizer_loader() -> SimpleNamespace:
    """Return an optimizer double without importing the optional DSPy dependency."""
    return SimpleNamespace(run_optimize=_raise_optimizer_error)


def _prepare_noop(_runs_dir: Path) -> None:
    """Leave a command's temporary runs directory empty before its failure."""


def _prepare_saved_report(runs_dir: Path) -> None:
    """Create the report presence check's required input artifact."""
    (runs_dir / "saved-run").mkdir()
    (runs_dir / "saved-run" / "report.json").touch()


def test_eval_keeps_stdout_factual_and_writes_artifacts(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    exit_code = main(
        [
            "eval",
            "--models",
            "stub",
            "--cases",
            "direct_turn_on_utility_room_ceiling",
            "--runs-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    lines = captured.out.splitlines()
    run_dir = Path(lines[0].removeprefix("run_dir: "))
    assert lines[0] == f"run_dir: {run_dir}"
    assert lines[1] == f"report_html: {run_dir / 'report.html'}"
    assert lines[2].startswith("quality_rate: ")
    assert lines[3].startswith("coverage_rate: ")
    assert lines[4].startswith("scored: ")
    # Machine KV stays ANSI-free and free of human Rich duplication.
    assert "\x1b" not in captured.out
    assert (run_dir / "report.json").is_file()
    assert (run_dir / "report.html").is_file()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_type"] == "llm_sandbox_eval_manifest"
    assert manifest["status"] == "complete"


def test_eval_token_telemetry_does_not_pollute_terminal_output(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configured: list[None] = []
    monkeypatch.setenv("LOGFIRE_TOKEN", "test-token")
    monkeypatch.setattr(logfire_config, "configure_logfire", lambda: configured.append(None))

    exit_code = main(
        [
            "eval",
            "--models",
            "stub",
            "--cases",
            "direct_turn_on_utility_room_ceiling",
            "--runs-dir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert configured == [None]
    assert captured.out.startswith("run_dir: ")
    assert "Logfire" not in captured.out
    assert "Logfire" not in captured.err


def test_eval_parser_does_not_accept_logfire_flag() -> None:
    with pytest.raises(SystemExit, match="2"):
        cli._build_parser().parse_args(["eval", "--logfire"])


@pytest.mark.parametrize(
    ("model_args", "expected_models"),
    [
        pytest.param([], ["stub"], id="default-stub"),
        pytest.param(
            [
                "--models",
                ",".join(
                    [
                        "gpt-5.4",
                        "anthropic:claude-sonnet-4-6",
                        "openai:gpt-5.4",
                        "openai-chat:gpt-4.1",
                        "openrouter:openai/gpt-5.4",
                        "stub",
                    ]
                ),
            ],
            [
                "openai-chat:gpt-5.4",
                "anthropic:claude-sonnet-4-6",
                "openai:gpt-5.4",
                "openai-chat:gpt-4.1",
                "openrouter:openai/gpt-5.4",
                "stub",
            ],
            id="explicit-models",
        ),
    ],
)
def test_eval_resolves_cli_model_ids_for_provider_and_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_args: list[str],
    expected_models: list[str],
) -> None:
    provider_models: list[str] = []

    def record_provider_model(model_id: str) -> Model:
        provider_models.append(model_id)
        return FunctionModel(_raise_provider_error, model_name=model_id)

    monkeypatch.setattr(agent_runner, "infer_model", record_provider_model)

    exit_code = main(
        [
            "eval",
            *model_args,
            "--cases",
            "direct_turn_on_utility_room_ceiling",
            "--runs-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert Counter(provider_models) == Counter(expected_models) - Counter(["stub"])
    [run_dir] = list(tmp_path.iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert [model["model_id"] for model in manifest["descriptor"]["models"]] == expected_models


def test_optimize_normalizes_only_cross_eval_model_ids(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured_configs: list[EvalConfig] = []

    def record_optimize(config: EvalConfig) -> SimpleNamespace:
        captured_configs.append(config)
        return SimpleNamespace(
            target_model=config.target_model,
            proposer_model=config.proposer_model,
            baseline_correct_rate=0.5,
            optimized_correct_rate=0.75,
            baseline_prompt_chars=100,
            optimized_prompt_chars=90,
            size_ratio=0.9,
            optimized_full_correct_rate=0.7,
            candidate_path=tmp_path / "optimize-run" / "optimized_candidate.json",
            cross_eval_run_dir=None,
        )

    monkeypatch.setattr(cli, "_load_optimizer", lambda: SimpleNamespace(run_optimize=record_optimize))

    target_model = "openrouter/openai/gpt-5.4"
    proposer_model = "anthropic/claude-sonnet-4-6"
    exit_code = main(
        [
            "optimize",
            "--target-model",
            target_model,
            "--proposer-model",
            proposer_model,
            "--cross-eval-models",
            ",".join(
                [
                    "gpt-5.4",
                    "anthropic:claude-sonnet-4-6",
                    "openai:gpt-5.4",
                    "openai-chat:gpt-4.1",
                    "openrouter:openai/gpt-5.4",
                    "stub",
                ]
            ),
            "--runs-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    [config] = captured_configs
    assert config.cross_eval_models == [
        "openai-chat:gpt-5.4",
        "anthropic:claude-sonnet-4-6",
        "openai:gpt-5.4",
        "openai-chat:gpt-4.1",
        "openrouter:openai/gpt-5.4",
        "stub",
    ]
    assert config.target_model == target_model
    assert config.proposer_model == proposer_model


def test_eval_reports_completed_cell_error_on_redirected_stderr(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(agent_runner, "make_model", _failing_model)

    exit_code = main(
        [
            "eval",
            "--models",
            "bad-model",
            "--cases",
            "direct_turn_on_utility_room_ceiling",
            "--runs-dir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    # Redirected stderr keeps deterministic KV lifecycle lines with the real operational cause.
    assert "cell_finished" in captured.err
    assert "incomplete·provider_error" in captured.err
    assert "matrix_started total=" in captured.err
    # stdout stays factual and free of provider error detail.
    assert "provider rejected model" not in captured.out


def test_escape_cancels_interactive_eval_writing_typed_partial_journal(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "_is_interactive_eval", lambda: True)
    monkeypatch.setattr(cli, "_EscapeWatcher", _ImmediateEscapeWatcher)

    exit_code = main(
        [
            "eval",
            "--models",
            "stub",
            "--cases",
            "direct_turn_on_utility_room_ceiling",
            "--runs-dir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 130
    # Non-zero exit keeps stdout empty; diagnostics and the partial journal go to stderr/disk.
    assert captured.out == ""
    assert "eval cancelled" in captured.err
    [run_dir] = list(tmp_path.iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "cancelled"
    # The typed partial journal is explicitly not a report (no report.json is written).
    assert (run_dir / "partial.json").is_file()
    assert not (run_dir / "report.json").exists()
    partial = reports.load_partial_artifact(run_dir / "partial.json")
    assert partial.artifact_type == "llm_sandbox_partial_run"
    assert partial.status == "cancelled"


@pytest.mark.parametrize(
    "matrix_failure",
    [
        pytest.param(_raise_keyboard_interrupt, id="keyboard-interrupt"),
        pytest.param(_raise_cancelled_error, id="cancelled-error"),
    ],
)
def test_eval_cancellation_writes_partial_journal_and_keeps_stdout_clean(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    matrix_failure: Callable[..., object],
) -> None:
    monkeypatch.setattr(cli, "_run_eval_matrix", matrix_failure)

    exit_code = main(["eval", "--models", "stub", "--runs-dir", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 130
    # Non-zero exit keeps stdout empty in all modes.
    assert captured.out == ""
    assert "eval cancelled" in captured.err
    assert "Traceback" not in captured.err
    # Cancellation persists the typed partial journal and a cancelled manifest.
    [run_dir] = list(tmp_path.iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "cancelled"
    partial = reports.load_partial_artifact(run_dir / "partial.json")
    assert partial.status == "cancelled"
    assert not (run_dir / "report.json").exists()


@pytest.mark.parametrize(
    ("command", "target", "target_name", "failure", "prepare", "expected_message"),
    [
        pytest.param(
            ["eval", "--models", "stub", "--runs-dir", "{runs_dir}"],
            reports,
            "write_report_json",
            _raise_artifact_error,
            _prepare_noop,
            "artifact write failed",
            id="eval-artifact-write",
        ),
        pytest.param(
            ["report", "saved-run", "--runs-dir", "{runs_dir}"],
            reports,
            "load_report",
            _raise_report_error,
            _prepare_saved_report,
            "report load failed",
            id="report-load",
        ),
        pytest.param(
            ["optimize", "--target-model", "model", "--runs-dir", "{runs_dir}"],
            cli,
            "_load_optimizer",
            _failing_optimizer_loader,
            _prepare_noop,
            "optimizer runtime failed",
            id="optimize-runtime",
        ),
    ],
)
def test_unexpected_command_failures_are_concise_and_keep_stdout_clean(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: list[str],
    target: object,
    target_name: str,
    failure: Callable[..., object],
    prepare: Callable[[Path], None],
    expected_message: str,
) -> None:
    prepare(tmp_path)
    monkeypatch.setattr(target, target_name, failure)

    exit_code = main([part.format(runs_dir=tmp_path) for part in command])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "RuntimeError" in captured.err
    assert expected_message in captured.err
    assert "Traceback" not in captured.err


def test_post_evaluation_write_failure_writes_failed_partial_preserving_records(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Fix #1: a report/HTML/manifest write failure after the matrix completes leaves a failed
    # manifest plus a failed partial journal that preserves the completed cells, with empty stdout.
    monkeypatch.setattr(reports, "write_report_json", _raise_artifact_error)

    exit_code = main(
        [
            "eval",
            "--models",
            "stub",
            "--cases",
            "direct_turn_on_utility_room_ceiling",
            "--runs-dir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    # Non-zero exit keeps stdout empty in all output modes.
    assert captured.out == ""
    assert "artifact write failed" in captured.err
    [run_dir] = list(tmp_path.iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    partial = reports.load_partial_artifact(run_dir / "partial.json")
    assert partial.status == "failed"
    assert partial.error is not None
    # The matrix completed before the write failed, so the journal preserves its records.
    assert partial.finished == 1
    assert partial.total == 1
    assert len(partial.records) == 1
    assert partial.records[0].trace.case_id == "direct_turn_on_utility_room_ceiling"
    # No report is written when the post-evaluation write fails.
    assert not (run_dir / "report.json").exists()


def test_cancellation_during_post_evaluation_persistence_writes_cancelled_journal(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An interruption during post-evaluation persistence leaves a cancelled manifest plus a
    # cancelled partial journal preserving the completed cells, exit 130, and empty stdout.
    monkeypatch.setattr(reports, "write_report_json", _raise_keyboard_interrupt_sync)

    exit_code = main(
        [
            "eval",
            "--models",
            "stub",
            "--cases",
            "direct_turn_on_utility_room_ceiling",
            "--runs-dir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 130
    assert captured.out == ""
    assert "eval cancelled" in captured.err
    [run_dir] = list(tmp_path.iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "cancelled"
    partial = reports.load_partial_artifact(run_dir / "partial.json")
    assert partial.status == "cancelled"
    # The matrix completed before the interruption, so the journal preserves its records.
    assert partial.finished == 1
    assert partial.total == 1
    assert len(partial.records) == 1
    assert partial.records[0].trace.case_id == "direct_turn_on_utility_room_ceiling"
    assert not (run_dir / "report.json").exists()


def test_debug_mode_reraises_unexpected_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Debug reraise still covers failures that escape _run_eval (e.g. the report subcommand).
    monkeypatch.setenv("LLM_SANDBOX_EVALS_DEBUG", "1")
    monkeypatch.setattr(reports, "load_report", _raise_report_error)
    _prepare_saved_report(tmp_path)

    with pytest.raises(RuntimeError, match="report load failed"):
        main(["report", "saved-run", "--runs-dir", str(tmp_path)])


@pytest.mark.parametrize("debug_value", [pytest.param("0", id="zero"), pytest.param("false", id="false")])
def test_non_debug_values_keep_unexpected_failures_concise(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    debug_value: str,
) -> None:
    monkeypatch.setenv("LLM_SANDBOX_EVALS_DEBUG", debug_value)
    monkeypatch.setattr(reports, "write_report_json", _raise_artifact_error)

    exit_code = main(["eval", "--models", "stub", "--runs-dir", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "artifact write failed" in captured.err
    assert "Traceback" not in captured.err


def test_main_preserves_argparse_system_exit() -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["eval", "--unknown-option"])


async def test_external_cancellation_stops_inner_interactive_matrix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocking_matrix(*_args: object, **_kwargs: object) -> Never:
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()
        raise AssertionError("unreachable")

    monkeypatch.setattr(cli.experiment, "run_matrix", blocking_matrix)
    monkeypatch.setattr(cli, "_is_interactive_eval", lambda: True)
    monkeypatch.setattr(cli, "_EscapeWatcher", lambda _cancel: nullcontext(SimpleNamespace(cancelled=False)))
    config = cli.EvalConfig(
        models=["stub"],
        candidates=["baseline"],
        prompt_profile="balanced",
        cases=None,
        homes=None,
        runs_dir=tmp_path,
    )
    outer = asyncio.create_task(
        cli._run_eval_matrix(config, "cancel-test", SimpleNamespace(handle=lambda _event: None))
    )
    await started.wait()

    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(outer, timeout=1)

    assert cancelled.is_set()


@pytest.mark.parametrize("interactive", [pytest.param(False, id="redirected"), pytest.param(True, id="interactive")])
async def test_run_eval_matrix_delivers_phase_callbacks_in_every_terminal_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, interactive: bool
) -> None:
    config = EvalConfig(
        models=["stub"],
        candidates=["baseline"],
        prompt_profile="balanced",
        cases=None,
        homes=None,
        runs_dir=tmp_path,
    )
    phase = LanePhaseEvent(MatrixCellRef("phase-case", "baseline", "stub", "home_minimal"), "thinking")
    delivered: list[LanePhaseEvent] = []

    async def emit_phase(
        _config: EvalConfig,
        *,
        descriptor: RunDescriptor | None = None,
        on_event: MatrixEventCallback | None = None,
        on_phase: LanePhaseCallback | None = None,
    ) -> EvaluationReport[MatrixCellRef, CaseTrace, MatrixCellMeta]:
        assert on_phase is not None
        on_phase(phase)
        return cast(EvaluationReport[MatrixCellRef, CaseTrace, MatrixCellMeta], None)

    monkeypatch.setattr(cli.experiment, "run_matrix", emit_phase)
    monkeypatch.setattr(cli, "_is_interactive_eval", lambda: interactive)
    monkeypatch.setattr(cli, "_EscapeWatcher", lambda _cancel: nullcontext(SimpleNamespace(cancelled=False)))

    await cli._run_eval_matrix(
        config,
        cli.experiment.build_run_descriptor(config, "phase-delivery", []),
        lambda _event: None,
        delivered.append,
    )

    assert delivered == [phase]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"scoring_version": 6, "cases": []}, id="v6-envelope"),
        pytest.param(
            {"scoring_version": 7, "cases": [{"output": {"scoring_version": 6}}]},
            id="v6-trace-in-v7-envelope",
        ),
    ],
)
def test_report_html_rejects_legacy_artifacts(
    payload: dict[str, object], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    (run_dir / "report.json").write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(["report", "legacy", "--html", "--runs-dir", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "legacy scoring artifact; rerun evaluation with scoring v7" in captured.err
    assert not (run_dir / "report.html").exists()


@pytest.mark.timeout(30)
def test_pty_sigint_interrupts_real_stub_eval_without_traceback_or_artifacts(tmp_path: Path) -> None:
    master_fd, slave_fd = pty.openpty()
    initial_settings = termios.tcgetattr(slave_fd)
    env = os.environ.copy()
    env.pop("LLM_SANDBOX_EVALS_DEBUG", None)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "llm_sandbox_evals",
            "eval",
            "--models",
            "stub",
            "--runs-dir",
            str(tmp_path),
        ],
        cwd=Path(__file__).parents[2],
        env=env,
        stdin=slave_fd,
        stdout=subprocess.PIPE,
        stderr=slave_fd,
    )
    os.set_blocking(master_fd, False)
    stderr = bytearray()
    try:
        deadline = time.monotonic() + 20
        while b"LLM Sandbox evaluation" not in stderr and time.monotonic() < deadline:
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if ready:
                stderr.extend(os.read(master_fd, 65536))
            if process.poll() is not None:
                break
        assert b"LLM Sandbox evaluation" in stderr
        process.send_signal(signal.SIGINT)
        interrupt_deadline = time.monotonic() + 10
        while process.poll() is None and time.monotonic() < interrupt_deadline:
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if ready:
                stderr.extend(os.read(master_fd, 65536))
        stdout, _ = process.communicate(timeout=1)
        while True:
            try:
                chunk = os.read(master_fd, 65536)
            except BlockingIOError:
                break
            if not chunk:
                break
            stderr.extend(chunk)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        restored_settings = termios.tcgetattr(slave_fd)
        os.close(master_fd)
        os.close(slave_fd)

    assert process.returncode == 130
    assert stdout == b""
    assert b"eval cancelled" in stderr
    assert b"Traceback" not in stderr
    assert restored_settings == initial_settings
    # Cancellation persists a typed partial journal and manifest rather than leaving nothing.
    [run_dir] = list(tmp_path.iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "cancelled"
    assert (run_dir / "partial.json").is_file()
    assert not (run_dir / "report.json").exists()


@pytest.mark.parametrize(
    ("continuation", "expected_cancellations"),
    [
        pytest.param((), 1, id="standalone-escape"),
        pytest.param((b"[A",), 0, id="arrow-sequence"),
        pytest.param((b"x",), 0, id="alt-sequence"),
    ],
)
async def test_escape_watcher_distinguishes_lone_escape_from_escape_sequences(
    continuation: tuple[bytes, ...], expected_cancellations: int
) -> None:
    cancellations: list[bool] = []
    watcher = cli._EscapeWatcher(lambda: cancellations.append(True))
    watcher._loop = asyncio.get_running_loop()

    watcher._handle_input(b"\x1b")
    for chunk in continuation:
        watcher._handle_input(chunk)
    await asyncio.sleep(cli._ESCAPE_DELAY_SECONDS * 2)

    assert len(cancellations) == expected_cancellations
    assert watcher.cancelled is (expected_cancellations == 1)
    watcher._cancel_pending_escape()


def _failing_model(_model_id: str) -> Model:
    """Return a deterministic provider failure for terminal diagnostic coverage."""
    return FunctionModel(_raise_provider_error, model_name="bad-model")


async def _raise_provider_error(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    """Simulate a provider failure without external transport."""
    raise RuntimeError("provider rejected model")


class _ImmediateEscapeWatcher:
    """Test seam that requests cancellation before the matrix task can run."""

    def __init__(self, on_escape: Callable[[], object]) -> None:
        """Store the active matrix cancellation callback."""
        self._on_escape = on_escape
        self.cancelled = False

    def __enter__(self) -> _ImmediateEscapeWatcher:
        """Request cancellation immediately, like a just-pressed Escape key."""
        self.cancelled = True
        self._on_escape()
        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        """Mirror the production watcher's non-suppressing context exit."""


@pytest.mark.parametrize(
    ("extra_args", "reason_key"),
    [
        pytest.param(["--cases", "no-such-case"], "unknown_case", id="unknown-case"),
        pytest.param(["--concurrency", "0"], "concurrency_invalid", id="bad-concurrency"),
        pytest.param(["--max-tool-calls", "0"], "max_tool_calls_invalid", id="bad-tool-cap"),
        pytest.param(["--model-timeout", "0"], "model_timeout_invalid", id="bad-timeout"),
        pytest.param(["--reasoning", "bogus"], "reasoning_invalid", id="bad-reasoning"),
        pytest.param(["--candidates", "no-such-candidate"], "candidates_unresolvable", id="bad-candidate"),
    ],
)
def test_eval_preflight_rejects_invalid_config_before_model_calls(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra_args: list[str],
    reason_key: str,
) -> None:
    model_calls: list[str] = []

    def _record_model(model_id: str) -> Model:
        model_calls.append(model_id)
        return FunctionModel(_raise_provider_error, model_name=model_id)

    # Scoped spy: if preflight fails to gate, the model factory would record a call.
    monkeypatch.setattr(agent_runner, "make_model", _record_model)

    exit_code = main(["eval", "--models", "stub", "--runs-dir", str(tmp_path), *extra_args])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert f"invalid_eval_config:{reason_key}" in captured.err
    # Validation happens before any run directory or model call is created.
    assert captured.out == ""
    assert not list(tmp_path.iterdir())
    assert model_calls == []


def test_eval_interactive_mode_emits_human_summary_on_stderr_and_keeps_stdout_clean(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Simulate an interactive TTY on stderr while stdin stays non-interactive, so the
    # non-interactive matrix path runs but the output mode resolves to human presentation.
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)

    exit_code = main(
        [
            "eval",
            "--models",
            "stub",
            "--cases",
            "direct_turn_on_utility_room_ceiling",
            "--runs-dir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    # Interactive mode never duplicates the machine KV block on stdout.
    assert captured.out == ""
    assert "quality_rate:" not in captured.out
    # The durable human final lands on stderr with the artifact location shown exactly once.
    assert captured.err.count("Artifacts:") == 1


def test_eval_machine_flag_forces_deterministic_kv_on_stdout_with_tty(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Even with stderr reporting a TTY, --machine forces deterministic KV on stdout.
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)

    exit_code = main(
        [
            "eval",
            "--models",
            "stub",
            "--cases",
            "direct_turn_on_utility_room_ceiling",
            "--runs-dir",
            str(tmp_path),
            "--machine",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.startswith("run_dir: ")
    assert "quality_rate: " in captured.out
    assert "coverage_rate: " in captured.out
    # Machine mode prints no human durable final on stderr.
    assert "Artifacts:" not in captured.err


def test_eval_failure_keeps_stdout_empty_and_writes_failed_partial_journal(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "_run_eval_matrix", _raise_runtime_error)

    exit_code = main(
        [
            "eval",
            "--models",
            "stub",
            "--cases",
            "direct_turn_on_utility_room_ceiling",
            "--runs-dir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    # Non-zero exit keeps stdout empty in all output modes.
    assert captured.out == ""
    [run_dir] = list(tmp_path.iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    partial = reports.load_partial_artifact(run_dir / "partial.json")
    assert partial.status == "failed"
    assert partial.error is not None


def test_manifest_descriptor_equals_report_experiment_metadata_including_created_at(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # Fix #7: one caller-owned descriptor drives both the manifest and the report's
    # experiment_metadata, so created_at and every field stay identical across artifacts.
    exit_code = main(
        [
            "eval",
            "--models",
            "stub",
            "--cases",
            "direct_turn_on_utility_room_ceiling",
            "--runs-dir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0

    run_dir = Path(captured.out.splitlines()[0].removeprefix("run_dir: "))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    report = reports.load_report(run_dir)

    manifest_descriptor = manifest["descriptor"]
    experiment_metadata = report.experiment_metadata
    # The created_at timestamp is the load-bearing field that proves a single descriptor instance.
    assert manifest_descriptor["created_at"] == experiment_metadata["created_at"]
    assert manifest_descriptor["run_id"] == experiment_metadata["run_id"]
    # The full descriptor payload is identical across the manifest and the persisted report.
    assert manifest_descriptor == experiment_metadata
