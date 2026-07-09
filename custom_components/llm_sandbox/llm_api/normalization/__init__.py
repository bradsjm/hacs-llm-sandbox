"""Fail-open AST normalization for execute_home_code."""

import ast

from .engine import RULES, RewriteEngine


def rewrite(code: str) -> tuple[str, tuple[str, ...]]:
    """Rewrite LLM code with shadow-safe forgiveness; fail open on any error."""
    try:
        tree = ast.parse(code)
    except Exception:  # noqa: BLE001 - fail-open normalization must catch all parser failures.
        return code, ()
    try:
        new_tree, labels = RewriteEngine(RULES).run(tree)
        if not labels:
            return code, ()
        ast.fix_missing_locations(new_tree)
        new_code = ast.unparse(new_tree)
        ast.parse(new_code)
    except Exception:  # noqa: BLE001 - fail-open normalization must catch all engine failures.
        return code, ()
    return new_code, labels
