import asyncio
from collections.abc import Callable
from pathlib import Path

from llm_sandbox_evals.cli import main
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
import pytest

from llm_sandbox_evals import agent_runner, cli


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
    assert lines[2].startswith("overall_mean: ")
    assert "\x1b" not in captured.out
    assert (run_dir / "report.json").is_file()
    assert (run_dir / "report.html").is_file()


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
    assert "current temperature in the living room" in captured.err
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
