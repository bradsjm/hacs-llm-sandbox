"""Outcome-evidence scoring for eval case results."""

import json
import re
from collections.abc import Iterable, Mapping, Sequence

from llm_sandbox_evals.schema import BlockedOutcome, CheckResult, EvalCase, ExpectedAction, ToolEvent, ToolResultCheck

_REGISTERED_TOOL_NAMES = frozenset(
    {
        "execute_home_code",
        "get_history",
        "get_logbook",
        "get_statistics",
    }
)
_GUIDANCE_PASSING_CONFIDENCE = frozenset({"exact", "high", "ambiguous"})
MAX_TOOL_CALLS = 10


def check_case(
    case: EvalCase,
    output: str,
    recorded_actions: tuple[dict[str, object], ...],
    tool_call_count: int,
    tool_events: tuple[ToolEvent, ...],
) -> list[CheckResult]:
    """Build deterministic outcome-evidence checks for one case."""
    checks: list[CheckResult] = []
    output_lower = output.lower()

    checks.append(_meaningful_oracle_check(case))

    answer_values = case.expected.answer_values + case.expected.expected_values

    # Final-answer text evidence is diagnostic only. LLM prose is intentionally
    # not a required scoring surface because small wording/formatting changes are
    # not reliable indicators of task success.
    missing_values = [value for value in answer_values if not _token_present(value, output_lower)]
    checks.append(
        CheckResult(
            name="answer_evidence_present",
            passed=not missing_values,
            required=False,
            feedback=f"missing={','.join(missing_values)}",
        )
    )

    structured_evidence = _structured_evidence_blob(tool_events, recorded_actions)
    provenance_lower = structured_evidence
    missing_provenance = [
        value for value in case.expected.provenance_values if not _token_present(value, provenance_lower)
    ]
    if case.expected.provenance_values:
        checks.append(
            CheckResult(
                name="provenance_evidence_present",
                passed=not missing_provenance,
                required=True,
                feedback=f"missing={','.join(missing_provenance)}",
            )
        )

    for index, tool_check in enumerate(case.expected.tool_result_checks):
        checks.append(_tool_result_check(tool_check, tool_events, index))

    if case.expected.answer_excludes:
        present_excludes = [fact for fact in case.expected.answer_excludes if fact.lower() in output_lower]
        checks.append(
            CheckResult(
                name="answer_excludes_absent",
                passed=not present_excludes,
                required=False,
                feedback=f"present={','.join(present_excludes)}",
            )
        )

    blocked_outcome = case.expected.blocked_outcome
    checks.append(_final_tool_ok_check(tool_events, allow_action_errors=blocked_outcome is not None))
    if blocked_outcome is None:
        checks.append(_actions_check(case.expected.actions, recorded_actions))
    else:
        checks.append(_blocked_outcome_check(blocked_outcome, recorded_actions))
    guidance_check = _guidance_quality_check(case.expected.guidance_candidate, tool_events)
    if guidance_check is not None:
        checks.append(guidance_check)
    checks.append(
        CheckResult(
            name="tool_calls_within_max",
            passed=tool_call_count <= MAX_TOOL_CALLS,
            required=True,
            feedback=f"calls={tool_call_count} max={MAX_TOOL_CALLS}",
        )
    )
    return checks


def score_case(checks: list[CheckResult]) -> float:
    """Score a case from required structured outcome gates."""
    if any(check.required and not check.passed for check in checks):
        return 0.0
    return 1.0


def mean_score(case_scores: list[float]) -> float:
    """Return the arithmetic mean for case scores."""
    if not case_scores:
        return 0.0
    return sum(case_scores) / len(case_scores)


def is_incomplete(checks: Iterable[CheckResult]) -> bool:
    """Return whether a trace failed due to provider/infra error, not candidate behavior.

    Incomplete cells (``model_error`` from timeouts or provider failures) are
    excluded from candidate/model mean-score denominators. ``tool_calls_exceeded``
    is a genuine model limit hit and is NOT incomplete.
    """
    return any(check.name == "model_error" for check in checks)


def _meaningful_oracle_check(case: EvalCase) -> CheckResult:
    """Return the required lint gate preventing empty/weak case oracles."""
    expected = case.expected
    has_oracle = bool(
        expected.provenance_values
        or expected.tool_result_checks
        or expected.actions
        or expected.blocked_outcome is not None
    )
    return CheckResult(
        name="meaningful_oracle",
        passed=has_oracle,
        required=True,
        feedback="ok" if has_oracle else "missing_answer_provenance_tool_action_or_blocked_expectation",
    )


def _structured_evidence_blob(
    tool_events: tuple[ToolEvent, ...], recorded_actions: tuple[dict[str, object], ...]
) -> str:
    """Return lowercased structured tool and action payloads for scoring evidence."""
    parts = [json.dumps(event.output, ensure_ascii=False, sort_keys=True, default=str) for event in tool_events] + [
        json.dumps(action, ensure_ascii=False, sort_keys=True, default=str) for action in recorded_actions
    ]
    return "\n".join(parts).lower()


def _token_present(token: str, evidence_lower: str) -> bool:
    r"""Return whether ``token`` appears with word boundaries where they apply.

    Word boundaries prevent short tokens like ``off`` from matching noise such as
    ``office`` while still matching a quoted state value like ``"off"``. Symbolic
    edge characters such as ``°`` cannot form ``\b`` boundaries, so boundary
    assertions are applied only on token edges that are word characters.
    """
    token_lower = token.lower()
    if not token_lower:
        return True
    prefix = r"\b" if _is_word_edge(token_lower[0]) else ""
    suffix = r"\b" if _is_word_edge(token_lower[-1]) else ""
    return re.search(prefix + re.escape(token_lower) + suffix, evidence_lower) is not None


def _is_word_edge(character: str) -> bool:
    """Return whether ``character`` participates in regex word boundaries."""
    return re.match(r"\w", character) is not None


def _final_tool_ok_check(tool_events: tuple[ToolEvent, ...], *, allow_action_errors: bool) -> CheckResult:
    """Return the required gate asserting the final tool event did not fail."""
    # Branch boundary: no tool events means the model answered without invoking a
    # tool. execution_ok passes vacuously; the evidence gate is responsible for
    # catching a non-accomplished task in that case.
    if not tool_events:
        return CheckResult(
            name="execution_ok",
            passed=True,
            required=True,
            feedback="no_tool_events",
        )
    error = _tool_event_error_kind(tool_events[-1], allow_action_errors)
    return CheckResult(
        name="execution_ok",
        passed=error is None,
        required=True,
        feedback="ok" if error is None else f"error={error}",
    )


def _tool_event_error_kind(event: ToolEvent, allow_action_errors: bool = False) -> str | None:
    """Return the error kind for a tool event, including unknown/hallucinated tools."""
    if event.tool_name not in _REGISTERED_TOOL_NAMES:
        return f"{event.tool_name}:unknown_tool"
    # Branch boundary: Pydantic AI may surface a hallucinated tool as an empty return.
    if not event.output:
        return f"{event.tool_name}:empty_output"
    error = _envelope_error_kind(event.output, allow_action_errors=allow_action_errors)
    if error is None:
        return None
    return f"{event.tool_name}:{error}"


def _envelope_error_kind(content: object, *, allow_action_errors: bool = False) -> str | None:
    """Return the error kind embedded in a tool return envelope, else None.

    Execute payloads carry ``execution.status`` (``ok`` when code ran) and may
    still include errored ``actions``. Recorder payloads carry a top-level
    ``status == "error"``. Any other non-empty shape is treated as success.
    """
    if not isinstance(content, Mapping):
        return None
    execution = content.get("execution")
    if isinstance(execution, Mapping):
        # Branch boundary: an execute envelope must end on status "ok".
        status = execution.get("status")
        if status != "ok":
            return str(status) if status is not None else "unknown"
        if not allow_action_errors:
            action_error = _action_error_kind(content)
            if action_error is not None:
                return action_error
        return None
    if content.get("status") == "error":
        error = content.get("error")
        if isinstance(error, Mapping):
            kind = error.get("key")
            return str(kind) if kind is not None else "error"
        return "error"
    if not allow_action_errors:
        action_error = _action_error_kind(content)
        if action_error is not None:
            return action_error
    return None


def _action_error_kind(content: Mapping[object, object]) -> str | None:
    """Return the first errored action key embedded in an execute envelope."""
    actions = content.get("actions")
    if not isinstance(actions, list):
        return None
    for action in actions:
        if not isinstance(action, Mapping) or action.get("status") != "error":
            continue
        error = action.get("error")
        if isinstance(error, Mapping):
            key = error.get("key")
            return f"action_error:{key}" if key is not None else "action_error"
        return "action_error"
    return None


def _guidance_quality_check(expected_candidate: str | None, tool_events: tuple[ToolEvent, ...]) -> CheckResult | None:
    """Return the optional gate requiring useful structured guidance on failures."""
    if expected_candidate is None:
        return None
    failing_events = [event for event in tool_events if _tool_event_error_kind(event) is not None]
    if not failing_events:
        return CheckResult(
            name="guidance_quality",
            passed=True,
            required=True,
            feedback=f"no_failure_for={expected_candidate}",
        )
    for event in failing_events:
        if _guidance_has_candidate(event.output, expected_candidate):
            return CheckResult(
                name="guidance_quality",
                passed=True,
                required=True,
                feedback=f"candidate={expected_candidate}",
            )
    return CheckResult(
        name="guidance_quality",
        passed=False,
        required=True,
        feedback=f"missing={expected_candidate}",
    )


def _guidance_has_candidate(content: Mapping[str, object], expected_candidate: str) -> bool:
    """Return whether a failing envelope/action guidance includes the expected candidate."""
    for guidance in _guidance_payloads(content):
        if guidance.get("confidence") not in _GUIDANCE_PASSING_CONFIDENCE:
            continue
        candidates = guidance.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            if candidate.get("id") == expected_candidate or candidate.get("name") == expected_candidate:
                return True
    return False


def _guidance_payloads(content: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Collect structured guidance payloads from execution, tool, and action errors."""
    payloads: list[Mapping[str, object]] = []
    execution = content.get("execution")
    if isinstance(execution, Mapping):
        guidance = execution.get("guidance")
        if isinstance(guidance, Mapping):
            payloads.append(guidance)
    error = content.get("error")
    if isinstance(error, Mapping):
        guidance = error.get("guidance")
        if isinstance(guidance, Mapping):
            payloads.append(guidance)
    actions = content.get("actions")
    if isinstance(actions, list):
        for action in actions:
            if not isinstance(action, Mapping):
                continue
            action_error = action.get("error")
            if not isinstance(action_error, Mapping):
                continue
            guidance = action_error.get("guidance")
            if isinstance(guidance, Mapping):
                payloads.append(guidance)
    return payloads


def _tool_result_check(expected: ToolResultCheck, tool_events: tuple[ToolEvent, ...], index: int) -> CheckResult:
    """Return a structured evidence check over successful recorder tool outputs."""
    matching_events = [
        event
        for event in tool_events
        if event.tool_name == expected.tool_name and _tool_event_error_kind(event) is None
    ]
    failures: list[str] = []
    if expected.tool_name == "execute_home_code" and matching_events:
        # Branch boundary: execute_home_code is often used as short discovery
        # glue, so one user outcome can be evidenced across multiple successful
        # snippets. Recorder tool checks remain per-call shape checks below.
        failures = _execute_result_failures(expected, _combined_execute_output(matching_events))
        if not failures:
            return CheckResult(
                name=f"tool_result_check_{index}",
                passed=True,
                required=True,
                feedback=f"tool={expected.tool_name}",
            )
    for event in matching_events:
        failures = _tool_result_failures(expected, event.output, event.args)
        if not failures:
            return CheckResult(
                name=f"tool_result_check_{index}",
                passed=True,
                required=True,
                feedback=f"tool={expected.tool_name}",
            )
    if not matching_events:
        failures.append("missing_successful_tool_result")
    return CheckResult(
        name=f"tool_result_check_{index}",
        passed=False,
        required=True,
        feedback=f"tool={expected.tool_name} failures={','.join(failures)}",
    )


def _tool_result_failures(
    expected: ToolResultCheck, output: Mapping[str, object], args: Mapping[str, object]
) -> list[str]:
    """Return structured recorder-result mismatches for one output payload."""
    if expected.tool_name == "execute_home_code":
        return _execute_result_failures(expected, output)
    if expected.tool_name == "get_history":
        return _history_result_failures(expected, output, args)
    if expected.tool_name == "get_statistics":
        return _statistics_result_failures(expected, output)
    if expected.tool_name == "get_logbook":
        return _logbook_result_failures(expected, output, args)
    return ["unsupported_tool"]


def _combined_execute_output(events: Sequence[ToolEvent]) -> dict[str, object]:
    """Return one searchable payload made from successful execute_home_code calls."""
    return {"execution": {"status": "ok"}, "output": [event.output for event in events]}


def _execute_result_failures(expected: ToolResultCheck, output: Mapping[str, object]) -> list[str]:
    """Return structured mismatches for one successful execute_home_code payload."""
    failures: list[str] = []
    result = output.get("output")
    printed = output.get("printed")
    has_result = result is not None or (isinstance(printed, list) and bool(printed))
    if expected.min_results == 0 and has_result:
        failures.append("unexpected_results")
    elif expected.min_results > 0 and not has_result:
        failures.append("empty_output")
    output_blob = json.dumps(output, ensure_ascii=False, sort_keys=True, default=str).lower()
    failures.extend(
        f"missing_entry_entity:{entity_id}"
        for entity_id in expected.entity_ids
        if not _token_present(entity_id, output_blob)
    )
    failures.extend(
        f"missing_entry_value:{value}" for value in expected.entry_values if not _token_present(value, output_blob)
    )
    return failures


def _history_result_failures(
    expected: ToolResultCheck, output: Mapping[str, object], args: Mapping[str, object]
) -> list[str]:
    rows = output.get("rows")
    if isinstance(rows, list):
        return _history_analytics_result_failures(expected, rows, args)

    failures: list[str] = []
    entities = output.get("entities")
    summary = output.get("summary")
    entity_map = entities if isinstance(entities, Mapping) else summary if isinstance(summary, Mapping) else {}
    for entity_id in expected.entity_ids:
        entity_payload = entity_map.get(entity_id) if isinstance(entity_map, Mapping) else None
        if not isinstance(entity_payload, Mapping):
            failures.append(f"missing_entity:{entity_id}")
            continue
        result_count = _result_count(entity_payload)
        if expected.min_results == 0 and result_count > 0:
            failures.append(f"unexpected_results:{entity_id}")
        elif expected.min_results > 0 and result_count < expected.min_results:
            failures.append(f"empty_entity:{entity_id}")
        entity_blob = json.dumps(entity_payload, ensure_ascii=False, sort_keys=True, default=str).lower()
        entry_values = (*expected.entry_values, *expected.entry_values_by_entity.get(entity_id, ()))
        failures.extend(
            f"missing_entry_value:{entity_id}:{value}"
            for value in entry_values
            if not _token_present(value, entity_blob)
        )
    return failures


def _history_analytics_result_failures(
    expected: ToolResultCheck, rows: list[object], args: Mapping[str, object]
) -> list[str]:
    """Return mismatches for declarative history analytics top-level rows."""
    failures: list[str] = []
    if expected.min_results == 0 and rows:
        failures.append("unexpected_results")
    elif len(rows) < expected.min_results:
        failures.append("empty_rows")
    rows_blob = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str).lower()
    requested_entity_ids = set(strings_from_value(args.get("entity_ids")))
    failures.extend(
        f"missing_row_entity:{entity_id}"
        for entity_id in expected.entity_ids
        if entity_id not in requested_entity_ids and not _token_present(entity_id, rows_blob)
    )
    if expected.entry_values_by_entity:
        for entity_id, entry_values in expected.entry_values_by_entity.items():
            entity_rows = [row for row in rows if isinstance(row, Mapping) and row.get("entity_id") == entity_id]
            entity_blob = json.dumps(entity_rows or rows, ensure_ascii=False, sort_keys=True, default=str).lower()
            failures.extend(
                f"missing_entry_value:{entity_id}:{value}"
                for value in entry_values
                if not _token_present(value, entity_blob)
            )
    else:
        failures.extend(
            f"missing_entry_value:{value}" for value in expected.entry_values if not _token_present(value, rows_blob)
        )
    return failures


def _statistics_result_failures(expected: ToolResultCheck, output: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    if expected.period is not None and output.get("period") != expected.period:
        failures.append(f"period:{output.get('period')}")
    statistics = output.get("statistics")
    if not isinstance(statistics, Mapping):
        return [*failures, "missing_statistics"]
    for statistic_id in expected.statistic_ids or expected.entity_ids:
        statistic_payload = statistics.get(statistic_id)
        if not isinstance(statistic_payload, Mapping):
            failures.append(f"missing_statistic:{statistic_id}")
            continue
        fields = statistic_payload.get("fields")
        if expected.fields and not set(expected.fields) <= set(strings_from_value(fields)):
            failures.append(f"fields:{statistic_id}")
        result_count = _result_count(statistic_payload)
        if expected.min_results == 0 and result_count > 0:
            failures.append(f"unexpected_results:{statistic_id}")
        elif expected.min_results > 0 and result_count < expected.min_results:
            failures.append(f"empty_statistic:{statistic_id}")
        statistic_blob = json.dumps(statistic_payload, ensure_ascii=False, sort_keys=True, default=str).lower()
        entry_values = (*expected.entry_values, *expected.entry_values_by_entity.get(statistic_id, ()))
        failures.extend(
            f"missing_entry_value:{statistic_id}:{value}"
            for value in entry_values
            if not _token_present(value, statistic_blob)
        )
    return failures


def _logbook_result_failures(
    expected: ToolResultCheck, output: Mapping[str, object], args: Mapping[str, object]
) -> list[str]:
    entries = output.get("entries")
    if not isinstance(entries, list):
        return ["missing_entries"]
    failures: list[str] = []
    if expected.min_results == 0 and entries:
        failures.append("unexpected_results")
    elif len(entries) < expected.min_results:
        failures.append("empty_entries")
    if expected.min_results > 0:
        failures.extend(
            f"missing_entry_entity:{entity_id}"
            for entity_id in expected.entity_ids
            if not any(isinstance(entry, Mapping) and entry.get("entity_id") == entity_id for entry in entries)
        )
    elif expected.entity_ids:
        # Branch boundary: empty logbook entries cannot prove entity scope. Require
        # directly verifiable queried entity IDs; selector-only empty queries do
        # not carry normalized entity provenance in ToolEvent args.
        if "entity_ids" not in args:
            failures.append("unverified_query_scope")
            return failures
        requested_entity_ids = set(strings_from_value(args.get("entity_ids")))
        failures.extend(
            f"missing_query_entity:{entity_id}"
            for entity_id in expected.entity_ids
            if entity_id not in requested_entity_ids
        )
    entries_blob = json.dumps(entries, sort_keys=True, default=str).lower()
    failures.extend(
        f"missing_entry_value:{value}" for value in expected.entry_values if not _token_present(value, entries_blob)
    )
    return failures


def _result_count(payload: Mapping[object, object]) -> int:
    """Return the authored result count for rows or summary count payloads."""
    rows = payload.get("rows")
    if isinstance(rows, list):
        return len(rows)
    state_counts = payload.get("state_counts")
    if isinstance(state_counts, Mapping):
        return len(state_counts)
    return 1 if payload else 0


def _actions_check(
    expected_actions: tuple[ExpectedAction, ...], recorded_actions: tuple[dict[str, object], ...]
) -> CheckResult:
    """Return the exact action side-effect check."""
    # Branch boundary: empty expected means no action is permitted (blocked cases).
    if not expected_actions:
        return CheckResult(
            name="actions_match",
            passed=not recorded_actions,
            required=True,
            feedback=f"unexpected={len(recorded_actions)}",
        )

    errored_actions = [action for action in recorded_actions if action.get("status") == "error"]
    expected_effects = _expected_effects(expected_actions)
    recorded_effects, duplicate_effects = _recorded_effects(recorded_actions)
    unmatched_expected = [effect for effect in expected_effects if effect not in recorded_effects]
    unexpected_recorded = [effect for effect in recorded_effects if effect not in expected_effects]
    target_mismatches = [
        effect
        for effect, expected_targets in expected_effects.items()
        if effect in recorded_effects and expected_targets is not None and recorded_effects[effect] != expected_targets
    ]
    passed = (
        not errored_actions
        and not duplicate_effects
        and not unmatched_expected
        and not unexpected_recorded
        and not target_mismatches
    )
    feedback_parts: list[str] = []
    if errored_actions:
        feedback_parts.append(f"errors={_format_recorded_actions(errored_actions)}")
    if duplicate_effects:
        feedback_parts.append(f"duplicates={','.join(duplicate_effects)}")
    if unmatched_expected:
        feedback_parts.append(f"unmatched={','.join(unmatched_expected)}")
    if unexpected_recorded:
        feedback_parts.append(f"extra={','.join(unexpected_recorded)}")
    if target_mismatches:
        feedback_parts.append(f"target_mismatch={','.join(target_mismatches)}")
    return CheckResult(
        name="actions_match",
        passed=passed,
        required=True,
        feedback=";".join(feedback_parts),
    )


def _blocked_outcome_check(expected: BlockedOutcome, recorded_actions: tuple[dict[str, object], ...]) -> CheckResult:
    """Return the structured blocked-action side-effect check."""
    blocked_actions = [action for action in recorded_actions if action.get("status") == "error"]
    successful_actions = [action for action in recorded_actions if action.get("status") != "error"]
    error_keys = [_recorded_action_error_key(action) for action in blocked_actions]
    disallowed_keys = [key for key in error_keys if expected.error_keys and key not in expected.error_keys]
    failures: list[str] = []
    if successful_actions:
        failures.append(f"successful_actions={_format_recorded_actions(successful_actions)}")
    if len(blocked_actions) > expected.max_attempts:
        failures.append(f"attempts={len(blocked_actions)} max={expected.max_attempts}")
    if disallowed_keys:
        failures.append(f"error_keys={','.join(disallowed_keys)}")
    return CheckResult(
        name="blocked_outcome",
        passed=not failures,
        required=True,
        feedback="ok" if not failures else ";".join(failures),
    )


def _expected_effects(expected_actions: tuple[ExpectedAction, ...]) -> dict[str, frozenset[str] | None]:
    """Aggregate expected effects by domain/service/service-data identity."""
    effects: dict[str, frozenset[str] | None] = {}
    mutable_targets: dict[str, set[str]] = {}
    for action in expected_actions:
        key = _expected_action_key(action)
        if not action.target_entity_ids:
            effects[key] = None
            continue
        targets = mutable_targets.setdefault(key, set())
        targets.update(action.target_entity_ids)
    for key, targets in mutable_targets.items():
        effects[key] = frozenset(targets)
    return effects


def _recorded_effects(
    recorded_actions: tuple[dict[str, object], ...],
) -> tuple[dict[str, frozenset[str] | None], list[str]]:
    """Aggregate successful recorded effects and flag repeated overlapping targets."""
    targets_by_key: dict[str, set[str]] = {}
    targetless_keys: set[str] = set()
    seen_targets_by_key: dict[str, set[str]] = {}
    duplicates: list[str] = []
    for action in recorded_actions:
        if action.get("status") == "error":
            continue
        key = _recorded_action_key(action)
        targets = frozenset(entity_ids_from_action(action))
        if not targets:
            if key in targetless_keys or key in seen_targets_by_key:
                _append_unique(duplicates, key)
            targetless_keys.add(key)
            continue
        seen_targets = seen_targets_by_key.setdefault(key, set())
        # Branch boundary: split calls may cover disjoint targets, but any
        # intersection means an entity was acted on more than once for this effect.
        if key in targetless_keys or seen_targets.intersection(targets):
            _append_unique(duplicates, key)
        seen_targets.update(targets)
        targets_by_key.setdefault(key, set()).update(targets)
    effects: dict[str, frozenset[str] | None] = {}
    for key in targetless_keys:
        effects[key] = None
    for key, grouped_targets in targets_by_key.items():
        effects[key] = frozenset(grouped_targets)
    return effects, duplicates


def _expected_action_key(action: ExpectedAction) -> str:
    """Return a stable effect key for an expected action."""
    return _action_key(action.domain, action.service, action.service_data)


def _recorded_action_key(action: Mapping[str, object]) -> str:
    """Return a stable effect key for a recorded action."""
    service_data = action.get("service_data")
    return _action_key(
        str(action.get("domain")),
        str(action.get("service")),
        service_data if isinstance(service_data, Mapping) else None,
    )


def _action_key(domain: str, service: str, service_data: Mapping[str, object] | None) -> str:
    """Return domain/service plus expected service-data values, excluding entity selectors."""
    comparable_data = _comparable_service_data(service_data)
    return f"{domain}.{service}:{json.dumps(comparable_data, sort_keys=True, default=str)}"


def _comparable_service_data(service_data: Mapping[str, object] | None) -> dict[str, object]:
    """Return service data relevant to effect matching, without target entity IDs."""
    if service_data is None:
        return {}
    return {
        key: _canonical_service_data_value(value)
        for key, value in service_data.items()
        if key not in {"entity_id", "entity_ids"}
    }


def _canonical_service_data_value(value: object) -> object:
    """Canonicalize numeric leaves while preserving JSON shape and non-numeric values."""
    # Branch boundary: bool is an int subclass, but service-data booleans must
    # remain exact booleans rather than numeric 0/1 equivalents.
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, list):
        return [_canonical_service_data_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonical_service_data_value(item) for key, item in value.items()}
    return value


def _recorded_action_error_key(action: Mapping[str, object]) -> str:
    """Return the stable error key from a blocked action record."""
    error = action.get("error")
    if isinstance(error, Mapping):
        key = error.get("key")
        if key is not None:
            return str(key)
    return "action_error"


def entity_ids_from_action(action: Mapping[str, object]) -> list[str]:
    """Resolve directly named entity ids from a recorded action mapping."""
    entity_ids: list[str] = []

    target = action.get("target")
    if isinstance(target, Mapping):
        entity_ids.extend(_entity_ids_from_mapping(target))

    service_data = action.get("service_data")
    if isinstance(service_data, Mapping):
        entity_ids.extend(_entity_ids_from_mapping(service_data))

    return _dedupe(entity_ids)


def _entity_ids_from_mapping(data: Mapping[str, object]) -> list[str]:
    """Resolve direct entity id fields from an action mapping."""
    entity_ids: list[str] = []
    entity_ids.extend(strings_from_value(data.get("entity_id")))
    entity_ids.extend(strings_from_value(data.get("entity_ids")))
    return _dedupe(entity_ids)


def strings_from_value(value: object) -> list[str]:
    """Return string(s) from scalar/list-like JSON values."""
    if isinstance(value, str):
        return [value]

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return [item for item in value if isinstance(item, str)]

    return []


def _format_recorded_actions(actions: Sequence[Mapping[str, object]]) -> str:
    """Return stable feedback for recorded actions that were not expected."""
    return ",".join(f"{action.get('domain')}.{action.get('service')}" for action in actions)


def _dedupe(entity_ids: list[str]) -> list[str]:
    """Preserve order while removing duplicates."""
    return list(dict.fromkeys(entity_ids))


def _append_unique(values: list[str], value: str) -> None:
    """Append one feedback value once while preserving first-seen order."""
    if value not in values:
        values.append(value)
