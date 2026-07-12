"""Versioned contracts for structured eval answers and scoring traces."""
# The models below are deliberately field-oriented public schema; class-level
# docstrings would add noise without improving the generated contract.
# ruff: noqa: D101, D102, D105, UP037

from dataclasses import dataclass, field
from math import isfinite
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

type JsonScalar = str | int | float | bool | None
type SubjectKind = Literal["entity", "device", "area", "automation", "repair", "notification", "service"]


class _Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ValueClaim(_Claim):
    kind: Literal["value"] = "value"
    subject_kind: SubjectKind
    subject_id: str = Field(min_length=1)
    field: Literal[
        "state",
        "attribute",
        "name",
        "manufacturer",
        "model",
        "enabled",
        "available",
        "service",
        "message",
        "status",
        "value",
        "unit",
    ]
    attribute_name: str | None = None
    value: JsonScalar

    @model_validator(mode="after")
    def validate_attribute_name(self) -> "ValueClaim":
        if (self.field == "attribute") != bool(self.attribute_name):
            raise ValueError("attribute_name is required only for attribute claims")
        return self


class RelationClaim(_Claim):
    kind: Literal["relation"] = "relation"
    subject_kind: Literal["entity", "device", "area", "automation"]
    subject_id: str = Field(min_length=1)
    relation: Literal[
        "entity_device", "entity_area", "device_area", "area_floor", "automation_target", "entity_service"
    ]
    object_kind: Literal["entity", "device", "area", "floor", "service"]
    object_id: str = Field(min_length=1)


class CollectionClaim(_Claim):
    kind: Literal["collection"] = "collection"
    collection: Literal["entity_ids", "automation_ids", "repair_ids", "notification_ids", "service_ids"]
    filter_kind: Literal["all", "area", "device", "floor", "label", "domain", "state"]
    filter_value: str | None = None
    items: list[str]

    @model_validator(mode="after")
    def validate_filter_and_items(self) -> "CollectionClaim":
        if (self.filter_kind == "all") != (self.filter_value is None):
            raise ValueError("all filters require null filter_value; other filters require a value")
        if any(not item for item in self.items) or self.items != sorted(set(self.items)):
            raise ValueError("collection items must be unique sorted nonempty strings")
        return self


class AggregateClaim(_Claim):
    kind: Literal["aggregate"] = "aggregate"
    source: Literal["states", "history", "statistics", "logbook"]
    operator: Literal[
        "count",
        "mean",
        "minimum",
        "maximum",
        "sum",
        "duration_seconds",
        "first_seen",
        "last_seen",
        "time_in_state",
        "convert",
    ]
    subject_ids: list[str]
    input_field: Literal["none", "state", "mean", "min", "max", "sum", "event_message"]
    input_value: JsonScalar = None
    value: str | int | float
    unit: str | None = None

    @model_validator(mode="after")
    def validate_inputs(self) -> "AggregateClaim":
        if any(not item for item in self.subject_ids) or self.subject_ids != sorted(set(self.subject_ids)):
            raise ValueError("aggregate subjects must be unique sorted nonempty strings")
        if (self.input_field == "none") != (self.input_value is None):
            raise ValueError("input_field none requires null input_value and vice versa")
        return self


class EventClaim(_Claim):
    kind: Literal["event"] = "event"
    source: Literal["logbook", "history", "automation_run"]
    entity_id: str = Field(min_length=1)
    event_kind: Literal["logbook_message", "state_transition", "automation_run"]
    value: str
    when: str | None = None


class NoDataClaim(_Claim):
    kind: Literal["no_data"] = "no_data"
    source: Literal["history", "statistics", "logbook", "automation"]
    scope_entity_ids: list[str]

    @model_validator(mode="after")
    def validate_scope(self) -> "NoDataClaim":
        if any(not item for item in self.scope_entity_ids) or self.scope_entity_ids != sorted(
            set(self.scope_entity_ids)
        ):
            raise ValueError("no-data scope must be unique sorted nonempty strings")
        return self


type AnswerClaim = Annotated[
    ValueClaim | RelationClaim | CollectionClaim | AggregateClaim | EventClaim | NoDataClaim,
    Field(discriminator="kind"),
]


class EvalAnswer(BaseModel):
    """The only model-produced eval result; answer prose is display-only."""

    model_config = ConfigDict(extra="forbid")
    answer: str
    claims: list[AnswerClaim] = Field(default_factory=list)


class ExpectedConclusion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    claim: AnswerClaim
    assertion: Literal["equals", "approximate", "exact_items", "contains_items", "empty"]
    tolerance: float | None = None

    @model_validator(mode="after")
    def validate_assertion(self) -> "ExpectedConclusion":
        kind = self.claim.kind
        if self.assertion == "equals" and kind not in {"value", "relation", "aggregate", "event"}:
            raise ValueError("equals is only valid for value, relation, aggregate, and event claims")
        if self.assertion == "approximate":
            if (
                not isinstance(self.claim, (ValueClaim, AggregateClaim))
                or not isinstance(self.claim.value, (int, float))
                or self.tolerance is None
            ):
                raise ValueError("approximate requires a numeric value or aggregate claim and tolerance")
            if not isfinite(self.tolerance) or self.tolerance <= 0:
                raise ValueError("approximate tolerance must be finite and positive")
        if self.assertion in {"exact_items", "contains_items"} and kind != "collection":
            raise ValueError("item assertions require a collection claim")
        if self.assertion == "empty" and (kind != "no_data" or self.tolerance is not None):
            raise ValueError("empty requires a no-data claim and no tolerance")
        if self.assertion != "approximate" and self.tolerance is not None:
            raise ValueError("tolerance is only valid for approximate assertions")
        return self


@dataclass(frozen=True, slots=True)
class PromptCandidate:
    id: str
    api_prompt: str
    execute_home_code_description: str
    get_history_description: str
    get_statistics_description: str
    get_logbook_description: str
    get_automation_description: str


@dataclass(frozen=True, slots=True)
class CaseContext:
    platform: str = "test"
    device_id: str | None = None
    language: str | None = "en"


@dataclass(frozen=True, slots=True)
class ExpectedAction:
    domain: str
    service: str
    target_entity_ids: tuple[str, ...] = ()
    service_data: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class BlockedOutcome:
    error_keys: tuple[str, ...] = ()
    actions: tuple[ExpectedAction, ...] = ()


@dataclass(frozen=True, slots=True)
class Expected:
    conclusions: tuple[ExpectedConclusion, ...] = ()
    actions: tuple[ExpectedAction, ...] = ()
    blocked_outcome: BlockedOutcome | None = None

    def __post_init__(self) -> None:
        if not self.conclusions and not self.actions and self.blocked_outcome is None:
            raise ValueError("expected oracle must declare conclusions, actions, or blocked_outcome")
        if self.blocked_outcome is not None and self.actions:
            raise ValueError("blocked cases cannot also declare allowed actions")
        if len({conclusion.model_dump_json() for conclusion in self.conclusions}) != len(self.conclusions):
            raise ValueError("duplicate expected conclusions are not allowed")


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    category: str
    home: str
    user_request: str
    actions_enabled: bool
    expected: Expected
    oracle_version: Literal[2] = 2
    llm_context: CaseContext = CaseContext()
    action_domains: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ToolEvent:
    tool_name: str
    args: dict[str, object]
    output: dict[str, object]
    call_index: int = 0
    turn_index: int = 0
    batch_index: int = 0
    batch_size: int = 1


@dataclass(frozen=True, slots=True)
class ConclusionResult:
    expected: ExpectedConclusion
    answer_claim: AnswerClaim | None
    semantic_status: Literal["matched", "mismatched"]
    grounding_status: Literal["grounded", "ungrounded"]
    reason: str


@dataclass(frozen=True, slots=True)
class ActionResult:
    status: Literal["allowed", "blocked", "no_action"]
    passed: bool
    mismatches: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ActionLedger:
    successful: tuple[dict[str, object], ...] = ()
    rejected: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class EvalDiagnostics:
    tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    execute_repairs: int = 0
    model_turns: int = 0
    parallel_batches: int = 0
    max_batch_size: int = 1
    elapsed_seconds: float | None = None
    cap_exhausted: bool = False
    usage: dict[str, int | float | None] | None = None
    failure: str | None = None


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    state: Literal["correct", "incorrect", "incomplete"]
    reason: str
    score: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", 1.0 if self.state == "correct" else 0.0)


@dataclass(frozen=True, slots=True)
class CaseTrace:
    case_id: str
    category: str
    candidate_id: str
    model_id: str
    answer: EvalAnswer | None
    expected: Expected
    outcome: CaseOutcome
    conclusions: tuple[ConclusionResult, ...]
    actions: tuple[ActionResult, ...]
    action_ledger: ActionLedger
    tool_events: tuple[ToolEvent, ...]
    diagnostics: EvalDiagnostics
    scoring_version: Literal[2] = 2
    provider_error: str | None = None
    user_request: str = ""
    conversation_id: str | None = None
