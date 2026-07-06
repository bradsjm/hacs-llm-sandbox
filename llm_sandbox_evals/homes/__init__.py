"""Frozen Home Assistant environment fixtures for evals.

Each fixture module exposes ``snapshot() -> HomeSnapshot`` (a fresh frozen
snapshot built from committed constants) and ``recorder() -> dict`` (canned
recorder rows). Fixtures never touch live Home Assistant.
"""

from types import ModuleType

from . import home_default, home_large, home_minimal, home_real


def get_home(name: str) -> ModuleType:
    """Return the fixture module for a home by name."""
    homes: dict[str, ModuleType] = {
        home_minimal.NAME: home_minimal,
        home_default.NAME: home_default,
        home_real.NAME: home_real,
        home_large.NAME: home_large,
    }
    # Unknown fixture names are caller errors; surface them without falling back.
    if name not in homes:
        raise KeyError(f"unknown home fixture: {name!r}")
    return homes[name]
