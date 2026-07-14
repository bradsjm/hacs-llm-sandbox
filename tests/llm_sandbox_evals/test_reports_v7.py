from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Literal

from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.experiment import run_matrix
from llm_sandbox_evals.reports import (
    _REPORT_ADAPTER,
    load_partial_artifact,
    load_report,
    rescore_trace,
    write_partial_artifact,
    write_report_json,
)
from llm_sandbox_evals.schema import (
    ActionLedger,
    ActionResult,
    CaseOutcome,
    CaseTrace,
    CompletedCellRecord,
    EvalDiagnostics,
    PartialRunArtifact,
    RequiredAction,
)
from pydantic import ValidationError
import pytest

from llm_sandbox_evals import reports


async def test_v7_report_round_trip_rescores_equivalent_target_partition(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = await run_matrix(config, run_id="v7-round-trip")
    original_answer = report.cases[0].output.answer
    run_dir = write_report_json(report, config, run_id="v7-round-trip-written")
    report_path = run_dir / "report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    output = payload["cases"][0]["output"]
    expected_action = RequiredAction(
        "light",
        "turn_on",
        ("light.utility_room_ceiling", "light.utility_room_accent"),
    )
    successful_actions = (
        {
            "domain": "light",
            "service": "turn_on",
            "target": {"entity_id": ["light.utility_room_ceiling"]},
            "service_data": {},
            "status": "success",
        },
        {
            "domain": "light",
            "service": "turn_on",
            "target": {"entity_id": ["light.utility_room_accent"]},
            "service_data": {},
            "status": "success",
        },
    )
    output["required_actions"] = [
        {
            "domain": expected_action.domain,
            "service": expected_action.service,
            "target_entity_ids": list(expected_action.target_entity_ids),
            "service_data": None,
        }
    ]
    output["outcome"] = {
        "state": "correct",
        "action_reason": "equivalent_target_partition",
        "score": 1.0,
    }
    output["action_result"] = {
        "passed": True,
        "reason": "equivalent_target_partition",
        "comparisons": [],
        "unexpected_actions": [],
    }
    output["action_ledger"] = {"successful": list(successful_actions), "rejected": []}
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    restored = load_report(run_dir).cases[0].output

    assert rescore_trace(restored) == CaseOutcome("correct", "equivalent_target_partition")
    assert restored.outcome == CaseOutcome("correct", "equivalent_target_partition")
    assert restored.required_actions == (expected_action,)
    assert restored.answer == original_answer
    assert restored.action_ledger == ActionLedger(successful=successful_actions)
    assert restored.action_result.reason == "equivalent_target_partition"
    assert restored.scoring_version == 7
    assert payload["scoring_version"] == 7
    assert output["scoring_version"] == 7


async def test_variant_fields_and_descriptor_survive_write_load(tmp_path: Path) -> None:
    config = EvalConfig(
        models=["stub"],
        candidates=["baseline"],
        prompt_profile=DEFAULT_PROMPT_PROFILE,
        cases=["direct_turn_on_utility_room_ceiling"],
        homes=None,
        runs_dir=tmp_path,
        reasoning_effort="high",
        temperature=0.7,
    )
    report = await run_matrix(config, run_id="variant-identity")
    run_dir = write_report_json(report, config, run_id="variant-identity-written")
    reloaded = load_report(run_dir)

    descriptor = reloaded.experiment_metadata
    models = descriptor["models"]
    assert isinstance(models, list)
    assert len(models) >= 1
    # Variant identity is persisted as structured fields with a derived display label.
    assert models[0]["model_id"] == "stub"
    assert models[0]["reasoning_effort"] == "high"
    assert models[0]["temperature"] == 0.7
    assert models[0]["variant_label"] == "stub(high)"
    cell = reloaded.cases[0].output
    assert cell.reasoning_effort == "high"
    assert cell.temperature == 0.7


@pytest.mark.parametrize("version", [None, 1, 5, 6], ids=["missing", "v1", "v5", "v6"])
async def test_legacy_and_invalid_artifacts_are_strictly_rejected(tmp_path: Path, version: int | None) -> None:
    config = _config(tmp_path)
    report = await run_matrix(config, run_id="artifact-version")
    run_dir = write_report_json(report, config, run_id=f"artifact-version-{version}")
    payload = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if version is None:
        del payload["scoring_version"]
    else:
        payload["scoring_version"] = version
    (run_dir / "report.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"^legacy scoring artifact; rerun evaluation with scoring v7$"):
        load_report(run_dir)


async def test_v6_trace_is_rejected_even_inside_a_v7_envelope(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = await run_matrix(config, run_id="legacy-trace")
    run_dir = write_report_json(report, config, run_id="legacy-trace-written")
    payload = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    payload["cases"][0]["output"]["scoring_version"] = 6
    (run_dir / "report.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"^legacy scoring artifact; rerun evaluation with scoring v7$"):
        load_report(run_dir)


def test_partial_journal_round_trips_and_is_not_report_shaped(tmp_path: Path) -> None:
    artifact = _partial_artifact(status="cancelled")
    path = write_partial_artifact(tmp_path / "partial.json", artifact)

    restored = load_partial_artifact(path)
    assert restored.artifact_type == "llm_sandbox_partial_run"
    assert restored.status == "cancelled"
    assert restored.finished == 1
    assert restored.total == 2
    assert restored.records[0].trace.case_id == "case-a"

    # The typed journal must never be interpretable as a native EvaluationReport.
    payload = json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(ValidationError):
        _REPORT_ADAPTER.validate_python(payload)


async def test_report_writer_cleans_temporary_file_when_json_serialization_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _config(tmp_path)
    report = await run_matrix(config, run_id="report-serialization-failure")
    run_id = "report-serialization-failure-written"
    run_dir = tmp_path / run_id
    monkeypatch.setattr(reports.json, "dumps", _raise_serialization_error)

    with pytest.raises(TypeError, match="cannot serialize artifact"):
        write_report_json(report, config, run_id=run_id)

    # A failed write exposes neither a target report nor a stranded atomic-write tempfile.
    assert not (run_dir / "report.json").exists()
    assert not list(run_dir.glob(".report.json.*"))


def test_partial_writer_cleans_temporary_file_when_json_serialization_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "partial.json"
    monkeypatch.setattr(reports.json, "dumps", _raise_serialization_error)

    with pytest.raises(TypeError, match="cannot serialize artifact"):
        write_partial_artifact(path, _partial_artifact(status="failed"))

    # The typed journal is either complete or absent; a serialization error leaves no partial file.
    assert not path.exists()
    assert not list(tmp_path.glob(".partial.json.*"))


def _config(runs_dir: Path) -> EvalConfig:
    return EvalConfig(
        models=["stub"],
        candidates=["baseline"],
        prompt_profile=DEFAULT_PROMPT_PROFILE,
        cases=["direct_turn_on_utility_room_ceiling"],
        homes=None,
        runs_dir=runs_dir,
    )


def _partial_artifact(*, status: Literal["cancelled", "failed"]) -> PartialRunArtifact:
    from llm_sandbox_evals.schema import ModelDescriptor, RunDescriptor

    trace = CaseTrace(
        case_id="case-a",
        candidate_id="baseline",
        model_id="stub",
        answer="Done.",
        required_actions=(RequiredAction("light", "turn_on", ("light.bedroom",)),),
        outcome=CaseOutcome("correct", "ok"),
        action_result=ActionResult(True, "ok"),
        action_ledger=ActionLedger(),
        tool_events=(),
        diagnostics=EvalDiagnostics(),
    )
    record = CompletedCellRecord(
        {
            "case_id": "case-a",
            "candidate_id": "baseline",
            "model_id": "stub",
            "home": "home_minimal",
            "reasoning_effort": None,
            "temperature": None,
        },
        trace,
        1,
        datetime.now(UTC).isoformat(),
    )
    return PartialRunArtifact(
        "llm_sandbox_partial_run",
        "partial-run",
        RunDescriptor(
            "partial-run",
            datetime.now(UTC).isoformat(),
            (ModelDescriptor("stub", None, None, "stub(default)"),),
            ("baseline",),
            ("case-a", "case-b"),
            DEFAULT_PROMPT_PROFILE,
            5,
            75.0,
            10,
        ),
        status,
        1,
        2,
        (record,),
        None,
        datetime.now(UTC).isoformat(),
    )


def _raise_serialization_error(*_args: object, **_kwargs: object) -> str:
    raise TypeError("cannot serialize artifact")
