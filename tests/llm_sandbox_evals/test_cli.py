from pathlib import Path

from llm_sandbox_evals.cli import main
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
import pytest

from llm_sandbox_evals import agent_runner


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


def _failing_model(_model_id: str) -> Model:
    """Return a deterministic provider failure for terminal diagnostic coverage."""
    return FunctionModel(_raise_provider_error, model_name="bad-model")


async def _raise_provider_error(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    """Simulate a provider failure without external transport."""
    raise RuntimeError("provider rejected model")
