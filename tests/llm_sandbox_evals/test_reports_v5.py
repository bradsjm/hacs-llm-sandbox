import json
from pathlib import Path

from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.experiment import run_matrix
from llm_sandbox_evals.reports import load_report, rescore_trace, write_report_json
import pytest


async def test_v5_report_round_trip_rescores_from_stored_ledger(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = await run_matrix(config, run_id="v5-round-trip")
    original = report.cases[0].output
    run_dir = write_report_json(report, config, run_id="v5-round-trip-written")

    restored = load_report(run_dir).cases[0].output

    assert rescore_trace(restored) == restored.outcome
    assert restored.expected_actions == original.expected_actions
    assert restored.answer == original.answer
    assert restored.action_ledger == original.action_ledger
    assert restored.action_result == original.action_result
    assert restored.action_result.reason == "ok"
    assert restored.action_result.comparisons[0].matched is True
    assert restored.action_result.comparisons[0].actual is not None
    assert restored.action_result.comparisons[0].actual.service == "turn_on"
    assert restored.scoring_version == 5
    payload = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["scoring_version"] == 5
    assert payload["cases"][0]["output"]["action_result"]["comparisons"][0]["service_matches"] is True


@pytest.mark.parametrize("version", [None, 1, 4], ids=["missing", "invalid", "v4"])
async def test_v4_and_invalid_artifacts_are_strictly_rejected(tmp_path: Path, version: int | None) -> None:
    config = _config(tmp_path)
    report = await run_matrix(config, run_id="artifact-version")
    run_dir = write_report_json(report, config, run_id=f"artifact-version-{version}")
    payload = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if version is None:
        del payload["scoring_version"]
    else:
        payload["scoring_version"] = version
    (run_dir / "report.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"^legacy scoring artifact; rerun evaluation$"):
        load_report(run_dir)


async def test_v4_trace_is_rejected_even_inside_a_v5_envelope(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = await run_matrix(config, run_id="v4-trace")
    run_dir = write_report_json(report, config, run_id="v4-trace-written")
    payload = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    payload["cases"][0]["output"]["scoring_version"] = 4
    (run_dir / "report.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"^legacy scoring artifact; rerun evaluation$"):
        load_report(run_dir)


def _config(runs_dir: Path) -> EvalConfig:
    return EvalConfig(
        models=["stub"],
        candidates=["baseline"],
        prompt_profile=DEFAULT_PROMPT_PROFILE,
        cases=["action_turn_on_bedroom_light"],
        homes=None,
        runs_dir=runs_dir,
    )
