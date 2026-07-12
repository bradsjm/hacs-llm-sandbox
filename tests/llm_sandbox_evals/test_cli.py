import asyncio
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
from typing import Never
import warnings

from llm_sandbox_evals.cli import main
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
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


def _raise_artifact_error(*_args: object, **_kwargs: object) -> object:
    """Raise a deterministic report artifact failure."""
    raise RuntimeError("artifact write failed")


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
            "state_living_temperature",
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
    assert lines[2].startswith("correct_rate: ")
    assert "\x1b" not in captured.out
    assert (run_dir / "report.json").is_file()
    assert (run_dir / "report.html").is_file()


def test_eval_native_output_mode_runs_through_cli_and_harness(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    exit_code = main(
        [
            "eval",
            "--models",
            "stub",
            "--cases",
            "state_living_temperature",
            "--output-mode",
            "json-schema",
            "--runs-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    run_dir = Path(captured.out.splitlines()[0].removeprefix("run_dir: "))
    assert (run_dir / "report.json").is_file()


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
            "state_living_temperature",
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
            "state_living_temperature",
            "--runs-dir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "cell finished" in captured.err
    assert "current living room temperature" in captured.err
    assert "provider rejected model" in captured.err
    assert "provider rejected model" not in captured.out


def test_escape_cancels_interactive_eval_without_artifacts_or_stdout(
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
            "state_living_temperature",
            "--runs-dir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 130
    assert captured.out == ""
    assert "eval cancelled" in captured.err
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("matrix_failure", "expected_message"),
    [
        pytest.param(_raise_keyboard_interrupt, "eval interrupted", id="keyboard-interrupt"),
        pytest.param(_raise_cancelled_error, "eval interrupted", id="cancelled-error"),
    ],
)
def test_eval_cancellation_is_clean_without_artifacts_or_stdout(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    matrix_failure: Callable[..., object],
    expected_message: str,
) -> None:
    monkeypatch.setattr(cli, "_run_eval_matrix", matrix_failure)

    exit_code = main(["eval", "--models", "stub", "--runs-dir", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 130
    assert captured.out == ""
    assert expected_message in captured.err
    assert "Traceback" not in captured.err
    assert not list(tmp_path.iterdir())


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


def test_debug_mode_reraises_unexpected_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLM_SANDBOX_EVALS_DEBUG", "1")
    monkeypatch.setattr(reports, "write_report_json", _raise_artifact_error)

    with pytest.raises(RuntimeError, match="artifact write failed"):
        main(["eval", "--models", "stub", "--runs-dir", str(tmp_path)])


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


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"cases": []}, id="missing-root-version"),
        pytest.param({"scoring_version": 2, "cases": [{"output": {}}]}, id="missing-trace-version"),
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
    assert "legacy scoring artifact" in captured.err
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
    assert b"eval interrupted" in stderr
    assert b"Traceback" not in stderr
    assert restored_settings == initial_settings
    assert not list(tmp_path.iterdir())


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
