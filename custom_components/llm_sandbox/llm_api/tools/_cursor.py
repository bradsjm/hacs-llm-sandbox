"""Opaque cursor pagination support for recorder-backed LLM tools.

Cursors carry the original query window plus a per-stream cutoff map (the ISO
timestamp of the oldest row returned on the previous page). They are base64-url
JSON, versioned, and unsigned by design: a cursor can only re-query a window
already validated at creation, visibility is re-checked against the fresh
snapshot on every call, and per-page budgets still apply, so a tampered cursor
cannot enlarge results or elevate privileges.
"""

import base64
import binascii
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import json

from homeassistant.util import dt as dt_util

from ..errors import RecoverableToolError

INVALID_CURSOR = "invalid_cursor"
_CURSOR_VERSION = 2
# Logbook is a flat stream; key its cutoff under a sentinel so all recorder
# tools share one cursor shape.
_LOGBOOK_CURSOR_KEY = "logbook"


@dataclass(frozen=True, slots=True)
class Cursor:
    """Decoded pagination cursor: query window plus per-stream cutoffs."""

    kind: str
    scope_ids: tuple[str, ...]
    start: datetime
    end: datetime
    cutoffs: dict[str, str]
    period: str | None = None
    statistic_types: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class _ValidatedCursorPayload:
    """Validated cursor fields before tool and scope matching."""

    kind: str
    scope_ids: tuple[str, ...]
    start: datetime
    end: datetime
    cutoffs: dict[str, str]
    period: str | None
    statistic_types: tuple[str, ...] | None


def encode_cursor(cursor: Cursor) -> str:
    """Encode a cursor to an opaque base64-url JSON string."""
    payload: dict[str, object] = {
        "v": _CURSOR_VERSION,
        "k": cursor.kind,
        "ids": list(cursor.scope_ids),
        "s": cursor.start.isoformat(),
        "e": cursor.end.isoformat(),
        "c": cursor.cutoffs,
    }
    if cursor.period is not None:
        payload["p"] = cursor.period
    if cursor.statistic_types is not None:
        payload["st"] = list(cursor.statistic_types)
    return base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")).decode(
        "ascii"
    )


def decode_cursor(value: object, *, expected_kind: str, expected_scope_ids: tuple[str, ...]) -> Cursor:
    """Decode an opaque cursor, raising ``invalid_cursor`` on malformation."""
    try:
        payload = _decode_cursor_payload(value)
        validated = _validate_cursor_payload(payload)
        _match_cursor_scope(validated, expected_kind=expected_kind, expected_scope_ids=expected_scope_ids)
        return _cursor_from_payload(validated)
    except ValueError, KeyError, TypeError, binascii.Error, UnicodeError, OverflowError:
        # Fail closed: any decode/version/shape problem restarts the sequence.
        raise RecoverableToolError(INVALID_CURSOR, {}) from None


def _decode_cursor_payload(value: object) -> dict[str, object]:
    """Decode base64 JSON without applying tool or scope matching."""
    if not isinstance(value, str):
        raise ValueError("cursor must be a string")
    decoded = json.loads(base64.urlsafe_b64decode(value.encode("ascii")))
    if not isinstance(decoded, dict):
        raise ValueError("cursor payload must be an object")
    return decoded


def _validate_cursor_payload(payload: dict[str, object]) -> _ValidatedCursorPayload:
    """Validate decoded cursor fields and normalize typed values."""
    if payload.get("v") != _CURSOR_VERSION:
        raise ValueError("unsupported cursor version")
    raw_kind = payload.get("k")
    if not isinstance(raw_kind, str):
        raise ValueError("invalid cursor kind")
    raw_scope_ids = payload.get("ids")
    if not isinstance(raw_scope_ids, list) or not all(isinstance(item, str) for item in raw_scope_ids):
        raise ValueError("invalid cursor scope")
    raw_start = dt_util.parse_datetime(str(payload["s"]))
    raw_end = dt_util.parse_datetime(str(payload["e"]))
    if raw_start is None or raw_end is None:
        raise ValueError("invalid cursor window")
    raw_cutoffs = payload.get("c", {})
    if not isinstance(raw_cutoffs, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in raw_cutoffs.items()
    ):
        raise ValueError("invalid cursor cutoffs")
    raw_period = payload.get("p")
    period = str(raw_period) if raw_period is not None else None
    raw_statistic_types = payload.get("st")
    statistic_types = None
    if raw_statistic_types is not None:
        if not isinstance(raw_statistic_types, list) or not all(isinstance(item, str) for item in raw_statistic_types):
            raise ValueError("invalid cursor statistic types")
        statistic_types = tuple(raw_statistic_types)
    return _ValidatedCursorPayload(
        kind=raw_kind,
        scope_ids=tuple(raw_scope_ids),
        start=dt_util.as_utc(raw_start),
        end=dt_util.as_utc(raw_end),
        cutoffs=dict(raw_cutoffs),
        period=period,
        statistic_types=statistic_types,
    )


def _match_cursor_scope(
    payload: _ValidatedCursorPayload, *, expected_kind: str, expected_scope_ids: tuple[str, ...]
) -> None:
    """Require a valid cursor to belong to this tool and resolved scope."""
    if payload.kind != expected_kind or payload.scope_ids != expected_scope_ids:
        raise ValueError("cursor scope mismatch")


def _cursor_from_payload(payload: _ValidatedCursorPayload) -> Cursor:
    """Construct the public cursor after validation and matching."""
    return Cursor(
        kind=payload.kind,
        scope_ids=payload.scope_ids,
        start=payload.start,
        end=payload.end,
        cutoffs=payload.cutoffs,
        period=payload.period,
        statistic_types=payload.statistic_types,
    )


def paginate_stream[T](
    rows: list[T],
    *,
    ts_of: Callable[[T], str],
    budget: int,
    cutoff_iso: str | None,
) -> tuple[list[T], str | None]:
    """Slice one ascending stream to one newest-first page.

    Keeps the newest ``budget`` rows strictly older than ``cutoff_iso``. The
    timestamp of the page's oldest row becomes the next resume boundary only
    when older rows still remain. Exact timestamp ties at a page boundary are
    excluded on the next page; adding a tiebreaker would overcomplicate these
    recorder streams where exact boundary ties are rare. The multi-stream pager
    passes the empty string ``""`` as the cutoff of an exhausted stream; since
    ``ts < ""`` is always false, that yields an empty page rather than a re-emit.
    """
    # Page > 1: drop everything at or newer than the resume boundary.
    remaining = [row for row in rows if ts_of(row) < cutoff_iso] if cutoff_iso is not None else list(rows)
    # Recorder streams are ascending, so the newest page is the tail.
    if budget > 0 and len(remaining) > budget:
        page = remaining[-budget:]
        more_remain = True
    else:
        page = remaining
        more_remain = False
    # Carry the oldest row in this page only while older rows remain.
    next_cutoff = ts_of(page[0]) if page and more_remain else None
    return page, next_cutoff
