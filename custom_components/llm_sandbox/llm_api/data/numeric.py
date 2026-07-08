"""Shared numeric coercion for SQLite rows and analytics."""

import math
from typing import cast


def finite_float(value: object) -> float | None:
    """Return a finite float for ``value`` or ``None`` when not coercible."""
    try:
        number = float(cast(str | int | float, value))
    except TypeError, ValueError:
        return None
    return number if math.isfinite(number) else None
