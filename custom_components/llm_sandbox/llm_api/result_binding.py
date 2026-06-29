"""Helpers for returning a Monty script's assigned result value."""

import ast

PROMOTED_LAST_EXPRESSION = "promoted_last_expression"


def _statement_blocks(statement: ast.stmt) -> list[list[ast.stmt]]:
    """Return nested statement blocks that can contain assignments."""
    blocks: list[list[ast.stmt]] = []
    if isinstance(statement, ast.If | ast.For | ast.AsyncFor | ast.While):
        blocks.extend([statement.body, statement.orelse])
    elif isinstance(statement, ast.With | ast.AsyncWith):
        blocks.append(statement.body)
    elif isinstance(statement, ast.Try):
        blocks.extend([statement.body, statement.orelse, statement.finalbody])
        blocks.extend(handler.body for handler in statement.handlers)
    elif isinstance(statement, ast.Match):
        blocks.extend(case.body for case in statement.cases)
    return blocks


def append_result_expression(code: str) -> str:
    """Append ``result`` when code assigns it anywhere at module scope."""
    if not _assigns_result(code):
        return code
    return f"{code.rstrip()}\nresult"


def promote_last_expression_to_result(code: str) -> tuple[str, list[str]]:
    """Rewrite a trailing bare expression at module scope to ``result = ...``.

    Returns the (possibly unchanged) code and a list of normalization labels.
    The transform only fires when the last top-level statement is a bare
    ``ast.Expr`` AND no explicit ``result = ...`` assignment exists anywhere
    at module scope (compound statements recurse; function/class bodies do
    not count, matching ``_assigns_result``). Any other shape (Assign,
    Import, FunctionDef, ...) is left untouched so we never silently drop a
    side effect or override an explicit binding. Fails open on SyntaxError.
    """
    if _assigns_result(code):
        return code, []
    try:
        module = ast.parse(code)
    except SyntaxError:
        return code, []
    if not module.body:
        return code, []
    last = module.body[-1]
    if not isinstance(last, ast.Expr):
        return code, []
    binding = ast.Assign(
        targets=[ast.Name(id="result", ctx=ast.Store())],
        value=last.value,
    )
    ast.copy_location(binding, last)
    module.body[-1] = binding
    ast.fix_missing_locations(module)
    return ast.unparse(module), [PROMOTED_LAST_EXPRESSION]


def _assigns_result(code: str) -> bool:
    try:
        module = ast.parse(code)
    except SyntaxError:
        return False
    return any(_statement_assigns_result(statement) for statement in module.body)


def _statement_assigns_result(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return False
    if isinstance(statement, ast.Assign):
        return any(isinstance(target, ast.Name) and target.id == "result" for target in statement.targets)
    if isinstance(statement, ast.AnnAssign | ast.AugAssign):
        return isinstance(statement.target, ast.Name) and statement.target.id == "result"
    for block in _statement_blocks(statement):
        for child in block:
            if _statement_assigns_result(child):
                return True
    return False
