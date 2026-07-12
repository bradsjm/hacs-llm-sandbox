"""Versioned contracts for concrete eval answers, expectations, and scoring traces."""
# The models below are deliberately field-oriented public schema; class-level
# docstrings would add noise without improving the generated contract.
# ruff: noqa: D101, D102, D105

from dataclasses import dataclass, field
from math import isfinite
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

type JsonScalar = str | int | float | bool | None


class _Expectation(BaseModel):
    """Internal authored oracle specification; models never emit it."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EntityExpectation(_Expectation):
    kind: Literal["entity"] = "entity"
    source: Literal["states", "history", "logbook", "automation"]
    entity_id: str = Field(min_length=1)
    input_field: Literal["state", "attribute", "name", "enabled", "message", "value", "run"]
    input_value: JsonScalar = None
    value: JsonScalar
    tolerance: float | None = None
    unit: str | None = None

    @model_validator(mode="after")
    def validate_source_field(self) -> EntityExpectation:
        allowed_fields = {
            "states": {"state", "attribute", "name"},
            "history": {"state", "attribute"},
            "logbook": {"message"},
            "automation": {"enabled", "name", "value", "run"},
        }
        if self.input_field not in allowed_fields[self.source]:
            raise ValueError(f"{self.input_field} is not available from {self.source} entity evidence")
        if self.input_field == "attribute":
            if not isinstance(self.input_value, str) or not self.input_value:
                raise ValueError("attribute expectations require a nonempty attribute name in input_value")
        elif self.input_value is not None:
            raise ValueError("input_value is only valid as the attribute name for attribute expectations")
        _validate_tolerance(self.value, self.tolerance)
        return self


class EntityCollectionExpectation(_Expectation):
    kind: Literal["entity_collection"] = "entity_collection"
    entity_ids: list[str]
    filter_kind: Literal["all", "area", "device", "floor", "label", "domain", "state"] | None = None
    filter_value: str | None = None

    @model_validator(mode="after")
    def validate_ids(self) -> EntityCollectionExpectation:
        _validate_sorted_ids(self.entity_ids, "collection entity ids")
        if (self.filter_kind in {None, "all"}) != (self.filter_value is None):
            raise ValueError("only a concrete collection filter accepts filter_value")
        return self


class AggregateExpectation(_Expectation):
    kind: Literal["aggregate"] = "aggregate"
    source: Literal["states", "history", "statistics", "logbook"]
    operator: Literal[
        "count",
        "mean",
        "minimum",
        "maximum",
        "sum",
        "duration_seconds",
        "time_in_state",
        "convert",
    ]
    subject_ids: list[str]
    input_field: Literal["none", "state", "mean", "min", "max", "sum", "event_message"]
    input_value: JsonScalar = None
    value: JsonScalar
    tolerance: float | None = None
    unit: str | None = None

    @model_validator(mode="after")
    def validate_inputs(self) -> AggregateExpectation:
        _validate_sorted_ids(self.subject_ids, "aggregate subjects")
        if (self.input_field == "none") != (self.input_value is None):
            raise ValueError("input_field none requires null input_value and vice versa")
        _validate_tolerance(self.value, self.tolerance)
        return self


class EntityRelationExpectation(_Expectation):
    kind: Literal["entity_relation"] = "entity_relation"
    relation: Literal["entity_device", "entity_area", "automation_target", "entity_service"]
    entity_id: str = Field(min_length=1)
    related_id: str = Field(min_length=1)
    scope_entity_ids: list[str] | None = None

    @model_validator(mode="after")
    def validate_scope(self) -> EntityRelationExpectation:
        if self.scope_entity_ids is not None:
            _validate_sorted_ids(self.scope_entity_ids, "relation scope")
        return self


class NoDataExpectation(_Expectation):
    kind: Literal["no_data"] = "no_data"
    source: Literal["history", "statistics", "logbook"]
    scope_entity_ids: list[str]

    @model_validator(mode="after")
    def validate_scope(self) -> NoDataExpectation:
        _validate_sorted_ids(self.scope_entity_ids, "no-data scope")
        return self


type AnswerExpectation = Annotated[
    EntityExpectation
    | EntityCollectionExpectation
    | AggregateExpectation
    | EntityRelationExpectation
    | NoDataExpectation,
    Field(discriminator="kind"),
]


def _validate_sorted_ids(values: list[str], label: str) -> None:
    if any(not item for item in values) or values != sorted(set(values)):
        raise ValueError(f"{label} must be unique sorted nonempty strings")


def _validate_tolerance(value: JsonScalar, tolerance: float | None) -> None:
    if tolerance is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance requires a numeric value and must be finite and positive")


class FinalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="Unrestricted display-only answer text; it is never parsed or scored.")


class EntityAnswer(FinalAnswer):
    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(description="Identifier of the entity whose value answers the request.")
    value: JsonScalar = Field(description="Observed scalar value that answers the request.")


class EntityCollectionAnswer(FinalAnswer):
    model_config = ConfigDict(extra="forbid")

    entity_ids: list[str] = Field(description="Entity identifiers comprising the requested collection.")


class AggregateAnswer(FinalAnswer):
    model_config = ConfigDict(extra="forbid")

    value: JsonScalar = Field(description="Computed scalar value that answers the request.")


class EntityRelationAnswer(FinalAnswer):
    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(description="Entity or automation identifier at the source of the relation.")
    related_id: str = Field(description="Identifier related to the source entity or automation.")


class NoDataAnswer(FinalAnswer):
    model_config = ConfigDict(extra="forbid")

    no_data: bool = Field(description="True when the requested scope produced no data.")


class ActionAnswer(FinalAnswer):
    model_config = ConfigDict(extra="forbid")


type AnswerShape = (
    EntityAnswer | EntityCollectionAnswer | AggregateAnswer | EntityRelationAnswer | NoDataAnswer | ActionAnswer
)


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
    expectation: AnswerExpectation | None = None
    actions: tuple[ExpectedAction, ...] = ()
    blocked_outcome: BlockedOutcome | None = None

    def __post_init__(self) -> None:
        if self.expectation is None and not self.actions and self.blocked_outcome is None:
            raise ValueError("expected oracle must declare an expectation, actions, or blocked_outcome")
        if self.blocked_outcome is not None and (self.actions or self.expectation is not None):
            raise ValueError("blocked cases cannot also declare an expectation or allowed actions")
        if (
            self.expectation is not None
            and self.actions
            and not isinstance(self.expectation, (EntityExpectation, AggregateExpectation))
        ):
            raise ValueError("conditional actions require an entity or aggregate expectation")


def select_answer_shape(expected: Expected) -> type[AnswerShape]:
    """Select the one concrete model-facing shape for an authored oracle."""
    registry: dict[type[_Expectation], type[AnswerShape]] = {
        EntityExpectation: EntityAnswer,
        EntityCollectionExpectation: EntityCollectionAnswer,
        AggregateExpectation: AggregateAnswer,
        EntityRelationExpectation: EntityRelationAnswer,
        NoDataExpectation: NoDataAnswer,
    }
    return ActionAnswer if expected.expectation is None else registry[type(expected.expectation)]


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    category: str
    home: str
    user_request: str
    actions_enabled: bool
    expected: Expected
    oracle_version: Literal[3] = 3
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
    expected: AnswerExpectation
    matched: bool
    grounded: bool
    reason: Literal["answer_mismatch", "evidence_missing", "ok"]


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
    answer: AnswerShape | None
    expected: Expected
    outcome: CaseOutcome
    conclusions: tuple[ConclusionResult, ...]
    actions: tuple[ActionResult, ...]
    action_ledger: ActionLedger
    tool_events: tuple[ToolEvent, ...]
    diagnostics: EvalDiagnostics
    scoring_version: Literal[4] = 4
    provider_error: str | None = None
    user_request: str = ""
    conversation_id: str | None = None
