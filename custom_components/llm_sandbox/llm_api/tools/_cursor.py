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
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from homeassistant.util import dt as dt_util

from ..errors import RecoverableToolError

INVALID_CURSOR = "invalid_cursor"
_CURSOR_VERSION = 1
# Logbook is a flat stream; key its cutoff under a sentinel so all recorder
# tools share one cursor shape.
_LOGBOOK_CURSOR_KEY = "logbook"


@dataclass(frozen=True, slots=True)
class Cursor:
    """Decoded pagination cursor: query window plus per-stream cutoffs."""

    start: datetime
    end: datetime
    cutoffs: dict[str, str]
    period: str | None = None
    statistic_types: tuple[str, ...] | None = None


def encode_cursor(cursor: Cursor) -> str:
    """Encode a cursor to an opaque base64-url JSON string."""
    payload: dict[str, object] = {
        "v": _CURSOR_VERSION,
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


def decode_cursor(value: object) -> Cursor:
    """Decode an opaque cursor, raising ``invalid_cursor`` on malformation."""
    try:
        if not isinstance(value, str):
            raise ValueError("cursor must be a string")
        obj = json.loads(base64.urlsafe_b64decode(value.encode("ascii")))
        if not isinstance(obj, dict) or obj.get("v") != _CURSOR_VERSION:
            raise ValueError("unsupported cursor version")
        raw_start = dt_util.parse_datetime(str(obj["s"]))
        raw_end = dt_util.parse_datetime(str(obj["e"]))
        # The cursor window was validated when created; None means corruption.
        if raw_start is None or raw_end is None:
            raise ValueError("invalid cursor window")
        start = dt_util.as_utc(raw_start)
        end = dt_util.as_utc(raw_end)
        raw_cutoffs = obj.get("c", {})
        if not isinstance(raw_cutoffs, dict):
            raise ValueError("invalid cursor cutoffs")
        cutoffs = {str(k): str(v) for k, v in raw_cutoffs.items()}
        raw_period = obj.get("p")
        period = str(raw_period) if raw_period is not None else None
        raw_statistic_types = obj.get("st")
        statistic_types = None
        if raw_statistic_types is not None:
            # Cursor-carried statistic field projection must stay a bounded string list.
            if not isinstance(raw_statistic_types, list) or not all(
                isinstance(item, str) for item in raw_statistic_types
            ):
                raise ValueError("invalid cursor statistic types")
            statistic_types = tuple(raw_statistic_types)
    except ValueError, KeyError, TypeError, binascii.Error:
        # Fail closed: any decode/version/shape problem restarts the sequence.
        raise RecoverableToolError(INVALID_CURSOR, {}) from None
    return Cursor(start=start, end=end, cutoffs=cutoffs, period=period, statistic_types=statistic_types)


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
    recorder streams where exact boundary ties are rare.
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
