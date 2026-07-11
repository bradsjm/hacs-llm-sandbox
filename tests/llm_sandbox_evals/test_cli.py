from pathlib import Path

from llm_sandbox_evals.cli import main
import pytest


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
