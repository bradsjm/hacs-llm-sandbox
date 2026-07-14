from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Literal

from custom_components.llm_sandbox.const import DEFAULT_PROMPT_PROFILE
from llm_sandbox_evals.config import EvalConfig
from llm_sandbox_evals.experiment import MatrixCellRef, run_matrix
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
    EndStateResult,
    EvalDiagnostics,
    ExecutionError,
    FailureClassification,
    PartialRunArtifact,
    RequiredAction,
)
from pydantic import ValidationError
from pydantic_evals.reporting import EvaluationReport, ReportCase
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
    output["desired_states"] = []
    output["overlay_state_seeds"] = []
    output["recorded_invocations"] = []
    output["end_state_result"] = {
        "status": "not_authored",
        "evaluable": False,
        "passed": False,
        "comparisons": [],
    }
    output["outcome"] = {
        "state": "correct",
        "scoring_mode": "actions",
        "score_reason": "equivalent_target_partition",
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

    assert rescore_trace(restored) == CaseOutcome("correct", "actions", "equivalent_target_partition")
    assert restored.outcome == CaseOutcome("correct", "actions", "equivalent_target_partition")
    assert restored.required_actions == (expected_action,)
    assert restored.answer == original_answer
    assert restored.action_ledger == ActionLedger(successful=successful_actions)
    assert restored.action_result.reason == "equivalent_target_partition"
    assert restored.scoring_version == 8
    assert payload["scoring_version"] == 8
    assert output["scoring_version"] == 8


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


async def test_execution_error_round_trips_and_defaults_for_older_v7_payload(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = await run_matrix(config, run_id="execution-error-round-trip")
    run_dir = write_report_json(report, config, run_id="execution-error-round-trip-written")
    report_path = run_dir / "report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    output = payload["cases"][0]["output"]
    expected = ExecutionError(
        exception_type="ModelHTTPError",
        message="provider request failed",
        status_code=503,
        provider_code="upstream_unavailable",
        provider_model="provider/model-v1",
        provider_detail='{"code":"upstream_unavailable"}',
    )
    output["execution_error"] = {
        "exception_type": expected.exception_type,
        "message": expected.message,
        "status_code": expected.status_code,
        "provider_code": expected.provider_code,
        "provider_model": expected.provider_model,
        "provider_detail": expected.provider_detail,
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_report(run_dir).cases[0].output.execution_error == expected

    del output["execution_error"]
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_report(run_dir).cases[0].output.execution_error is None


async def test_report_writer_creates_empty_error_log_for_successful_stub_report(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = await run_matrix(config, run_id="empty-error-log")

    run_dir = write_report_json(report, config, run_id="empty-error-log-written")

    assert (run_dir / "report.json").is_file()
    assert (run_dir / "errors.log").read_bytes() == b""


def test_report_writer_preserves_each_incomplete_error_as_ordered_ndjson(tmp_path: Path) -> None:
    provider_detail = json.dumps(
        {
            "error": {
                "code": "token_quota_exceeded",
                "message": "Daily quota exhausted; retry tomorrow.",
            }
        }
    )
    full_rate_limit_detail = (
        "Traceback (most recent call last):\n"
        "ModelHTTPError: status_code=429 code=token_quota_exceeded\n"
        f"provider body: {provider_detail}"
    )
    repeated_rate_limit = _incomplete_trace(
        case_id="rate-limit-case",
        candidate_id="baseline",
        model_id="openrouter:cerebras/llama-3.3-70b",
        reasoning_effort="high",
        temperature=0.4,
        conversation_id="conversation-rate-limit",
        user_request="Turn on the Utility Room ceiling light.",
        classification="rate_limit",
        provider_error=full_rate_limit_detail,
        execution_error=ExecutionError(
            exception_type="ModelHTTPError",
            message="Provider rate limit exceeded",
            status_code=429,
            provider_code="token_quota_exceeded",
            provider_model="cerebras/llama-3.3-70b",
            provider_detail=provider_detail,
        ),
    )
    provider_only_detail = "Provider closed the request before structured metadata was captured."
    provider_only = _incomplete_trace(
        case_id="provider-only-case",
        candidate_id="candidate-b",
        model_id="anthropic:claude-sonnet-4-6",
        reasoning_effort=None,
        temperature=None,
        conversation_id=None,
        user_request="Toggle the Utility Room outlet.",
        classification="provider_error",
        provider_error=provider_only_detail,
        execution_error=None,
    )
    report = _report_with_traces(repeated_rate_limit, repeated_rate_limit, provider_only)
    config = _config(tmp_path)

    run_dir = write_report_json(report, config, run_id="ordered-errors")

    physical_lines = (run_dir / "errors.log").read_bytes().splitlines()
    records = [json.loads(line) for line in physical_lines]
    rate_limit_record = {
        "classification": "rate_limit",
        "case_id": "rate-limit-case",
        "candidate_id": "baseline",
        "model_id": "openrouter:cerebras/llama-3.3-70b",
        "variant": "openrouter:cerebras/llama-3.3-70b(high)",
        "reasoning_effort": "high",
        "temperature": 0.4,
        "conversation_id": "conversation-rate-limit",
        "user_request": "Turn on the Utility Room ceiling light.",
        "exception_type": "ModelHTTPError",
        "message": "Provider rate limit exceeded",
        "status_code": 429,
        "provider_code": "token_quota_exceeded",
        "provider_model": "cerebras/llama-3.3-70b",
        "provider_detail": provider_detail,
        "detail": full_rate_limit_detail,
    }
    provider_only_record = {
        "classification": "provider_error",
        "case_id": "provider-only-case",
        "candidate_id": "candidate-b",
        "model_id": "anthropic:claude-sonnet-4-6",
        "variant": "anthropic:claude-sonnet-4-6(default)",
        "reasoning_effort": None,
        "temperature": None,
        "conversation_id": None,
        "user_request": "Toggle the Utility Room outlet.",
        "exception_type": None,
        "message": None,
        "status_code": None,
        "provider_code": None,
        "provider_model": None,
        "provider_detail": None,
        "detail": provider_only_detail,
    }
    assert records == [rate_limit_record, rate_limit_record, provider_only_record]
    assert physical_lines[0] == physical_lines[1]


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

    with pytest.raises(ValueError, match=r"^legacy scoring artifact; rerun evaluation with scoring v8$"):
        load_report(run_dir)


async def test_v6_trace_is_rejected_even_inside_a_v7_envelope(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = await run_matrix(config, run_id="legacy-trace")
    run_dir = write_report_json(report, config, run_id="legacy-trace-written")
    payload = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    payload["cases"][0]["output"]["scoring_version"] = 6
    (run_dir / "report.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"^legacy scoring artifact; rerun evaluation with scoring v8$"):
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
    run_dir.mkdir()
    errors_path = run_dir / "errors.log"
    report_path = run_dir / "report.json"
    errors_path.write_bytes(b"existing errors\n")
    report_path.write_bytes(b"existing report\n")
    monkeypatch.setattr(reports.json, "dumps", _raise_serialization_error)

    with pytest.raises(TypeError, match="cannot serialize artifact"):
        write_report_json(report, config, run_id=run_id)

    # Serialization completes before either independently atomic target can be replaced.
    assert errors_path.read_bytes() == b"existing errors\n"
    assert report_path.read_bytes() == b"existing report\n"
    assert not list(run_dir.glob(".errors.log.*"))
    assert not list(run_dir.glob(".report.json.*"))


async def test_error_log_replace_failure_prevents_report_and_cleans_temporary_file(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = await run_matrix(config, run_id="error-log-replace-failure")
    run_dir = tmp_path / "error-log-replace-failure-written"
    errors_path = run_dir / "errors.log"
    errors_path.mkdir(parents=True)

    with pytest.raises(IsADirectoryError):
        write_report_json(report, config, run_id=run_dir.name)

    assert errors_path.is_dir()
    assert not (run_dir / "report.json").exists()
    assert not list(run_dir.glob(".errors.log.*"))
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


def _incomplete_trace(
    *,
    case_id: str,
    candidate_id: str,
    model_id: str,
    reasoning_effort: str | None,
    temperature: float | None,
    conversation_id: str | None,
    user_request: str,
    classification: FailureClassification,
    provider_error: str,
    execution_error: ExecutionError | None,
) -> CaseTrace:
    return CaseTrace(
        case_id=case_id,
        candidate_id=candidate_id,
        model_id=model_id,
        answer=None,
        required_actions=(RequiredAction("light", "turn_on", ("light.utility_room_ceiling",)),),
        desired_states=(),
        overlay_state_seeds=(),
        recorded_invocations=(),
        end_state_result=EndStateResult("not_authored", False, False),
        outcome=CaseOutcome("incomplete", None, None),
        action_result=ActionResult(False, "missing_action"),
        action_ledger=ActionLedger(),
        tool_events=(),
        diagnostics=EvalDiagnostics(failure=classification),
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        provider_error=provider_error,
        execution_error=execution_error,
        user_request=user_request,
        conversation_id=conversation_id,
    )


def _report_with_traces(*traces: CaseTrace) -> reports.MatrixReport:
    cases = []
    for trace in traces:
        cell = MatrixCellRef(
            trace.case_id,
            trace.candidate_id,
            trace.model_id,
            "home_minimal",
            trace.reasoning_effort,
            trace.temperature,
        )
        cases.append(
            ReportCase(
                name=f"{trace.candidate_id}/{trace.model_id}/{trace.case_id}",
                inputs=cell,
                metadata={
                    "run_id": "ordered-errors",
                    "case_id": trace.case_id,
                    "candidate_id": trace.candidate_id,
                    "model_id": trace.model_id,
                    "home": cell.home,
                    "reasoning_effort": trace.reasoning_effort,
                    "temperature": trace.temperature,
                },
                expected_output=None,
                output=trace,
                metrics={},
                attributes={},
                scores={},
                labels={},
                assertions={},
                task_duration=0.0,
                total_duration=0.0,
            )
        )
    return EvaluationReport(name="ordered-errors", cases=cases)


def _partial_artifact(*, status: Literal["cancelled", "failed"]) -> PartialRunArtifact:
    from llm_sandbox_evals.schema import ModelDescriptor, RunDescriptor

    trace = CaseTrace(
        case_id="case-a",
        candidate_id="baseline",
        model_id="stub",
        answer="Done.",
        required_actions=(RequiredAction("light", "turn_on", ("light.bedroom",)),),
        desired_states=(),
        overlay_state_seeds=(),
        recorded_invocations=(),
        end_state_result=EndStateResult("not_authored", False, False),
        outcome=CaseOutcome("correct", "actions", "ok"),
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
