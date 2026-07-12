"""Exact successful and rejected action ledgers."""
# ruff: noqa: D103

from collections.abc import Mapping, Sequence
import json

from llm_sandbox_evals.schema import ActionLedger, ActionResult, BlockedOutcome, ExpectedAction


def build_action_ledger(actions: Sequence[Mapping[str, object]]) -> ActionLedger:
    copied = tuple(dict(action) for action in actions)
    return ActionLedger(
        tuple(action for action in copied if action.get("status") != "error"),
        tuple(action for action in copied if action.get("status") == "error"),
    )


def score_actions(
    expected: tuple[ExpectedAction, ...], blocked: BlockedOutcome | None, ledger: ActionLedger
) -> ActionResult | None:
    if blocked is not None:
        mismatches: list[str] = []
        if ledger.successful:
            mismatches.append("successful_effect")
        wanted = {_effect_key(action): _targets(action.target_entity_ids) for action in blocked.actions}
        actual = _ledger_effects(ledger.rejected, mismatches)
        if not ledger.rejected:
            mismatches.append("missing_rejected_effect")
        expected_keys = set(blocked.error_keys)
        actual_keys = {_error_key(action) for action in ledger.rejected}
        if expected_keys != actual_keys:
            if expected_keys - actual_keys:
                mismatches.append("missing_error_key")
            if actual_keys - expected_keys:
                mismatches.append("unexpected_error_key")
        mismatches.extend(key for key in wanted if key not in actual)
        mismatches.extend(key for key in actual if key not in wanted)
        for key, targets in wanted.items():
            if key in actual and actual[key] != targets:
                mismatches.append(f"target:{key}")
        return ActionResult("blocked", not mismatches, tuple(dict.fromkeys(mismatches)))
    if not expected:
        return None
    mismatches = []
    wanted = {_effect_key(action): _targets(action.target_entity_ids) for action in expected}
    actual = _ledger_effects(ledger.successful, mismatches)
    mismatches.extend(key for key in wanted if key not in actual)
    mismatches.extend(key for key in actual if key not in wanted)
    for key, targets in wanted.items():
        if key in actual and actual[key] != targets:
            mismatches.append(f"target:{key}")
    return ActionResult("allowed", not mismatches, tuple(dict.fromkeys(mismatches)))


def _ledger_effects(
    actions: Sequence[Mapping[str, object]], mismatches: list[str]
) -> dict[str, frozenset[str] | None]:
    result: dict[str, frozenset[str] | None] = {}
    seen: dict[str, set[str]] = {}
    for action in actions:
        key = _recorded_key(action)
        action_targets = _action_targets(action)
        if len(action_targets) != len(set(action_targets)):
            mismatches.append(f"duplicate:{key}")
        targets = set(action_targets)
        if key in result and (result[key] is None or not targets.isdisjoint(seen.get(key, set()))):
            mismatches.append(f"duplicate:{key}")
        if not targets:
            if key in result:
                mismatches.append(f"duplicate:{key}")
            result[key] = None
        else:
            seen.setdefault(key, set()).update(targets)
            result[key] = frozenset(seen[key])
    return result


def _effect_key(action: ExpectedAction) -> str:
    return _key(action.domain, action.service, action.service_data)


def _recorded_key(action: Mapping[str, object]) -> str:
    domain = str(action.get("domain", ""))
    service = str(action.get("service", ""))
    if "." in service and not domain:
        domain, service = service.split(".", 1)
    data = action.get("service_data")
    return _key(domain, service, data if isinstance(data, Mapping) else None)


def _key(domain: str, service: str, data: Mapping[str, object] | None) -> str:
    comparable = {
        key: _canonical(value) for key, value in (data or {}).items() if key not in {"entity_id", "entity_ids"}
    }
    return f"{domain}.{service}:{json.dumps(comparable, sort_keys=True)}"


def _canonical(value: object) -> object:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    return value


def _action_targets(action: Mapping[str, object]) -> tuple[str, ...]:
    values: list[str] = []
    for container in (action.get("target"), action.get("service_data")):
        if isinstance(container, Mapping):
            for key in ("entity_id", "entity_ids"):
                value = container.get(key)
                if isinstance(value, str):
                    values.append(value)
                elif isinstance(value, list):
                    values.extend(item for item in value if isinstance(item, str))
    return tuple(values)


def _targets(values: tuple[str, ...]) -> frozenset[str] | None:
    return frozenset(values) if values else None


def _error_key(action: Mapping[str, object]) -> str:
    error = action.get("error")
    return str(error.get("key", "action_error")) if isinstance(error, Mapping) else "action_error"
