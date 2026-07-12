import json
from pathlib import Path

from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.experiment import run_matrix
from llm_sandbox_evals.reports import load_report, rescore_trace, write_report_json
import pytest


async def test_v2_report_round_trip_rescores_from_stored_trace(tmp_path: Path) -> None:
    config = EvalConfig(
        models=["stub"],
        candidates=["baseline"],
        prompt_profile=DEFAULT_PROMPT_PROFILE,
        cases=["state_living_temperature"],
        homes=None,
        runs_dir=tmp_path,
    )
    report = await run_matrix(config, run_id="v2-round-trip")
    original = report.cases[0].output
    run_dir = write_report_json(report, config, run_id="v2-round-trip-written")

    loaded = load_report(run_dir)
    restored = loaded.cases[0].output
    rescored = rescore_trace(restored)

    assert rescored == restored.outcome
    assert restored.expected == original.expected
    assert restored.answer == original.answer
    assert restored.tool_events == original.tool_events
    assert restored.action_ledger == original.action_ledger
    assert json.loads((run_dir / "report.json").read_text(encoding="utf-8"))["scoring_version"] == 2


def test_legacy_or_incomplete_report_is_rejected_before_validation(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "scoring_version": 2,
                "cases": [{"output": {"answer": None, "expected": {}, "outcome": {"state": "incorrect"}}}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"^legacy scoring artifact; rerun evaluation$"):
        load_report(run_dir)


def test_legacy_report_without_scoring_version_is_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy-without-version"
    run_dir.mkdir()
    (run_dir / "report.json").write_text(json.dumps({"cases": []}), encoding="utf-8")

    with pytest.raises(ValueError, match=r"^legacy scoring artifact; rerun evaluation$"):
        load_report(run_dir)
