"""Failure context contracts for pure recovery guidance."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType


class Intent(StrEnum):
    """Recovery intent categories understood by the guidance engine."""

    READ_STATE = "read_state"
    CALL_SERVICE = "call_service"
    RESOLVE_SELECTOR = "resolve_selector"
    QUERY_HISTORY = "query_history"
    SQL_TABLE = "sql_table"
    SQL_COLUMN = "sql_column"
    CAPTURE_IMAGE = "capture_image"
    CODE_NAME = "code_name"
    CODE_ATTRIBUTE = "code_attribute"


@dataclass(frozen=True, slots=True)
class FailureContext:
    """Frozen description of a failed literal and the surface it was used against."""

    intent: Intent
    requested: str
    domain: str = ""
    service: str = ""
    selector: str = ""
    service_data: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    available_attributes: tuple[str, ...] = ()
    table_name: str = ""
