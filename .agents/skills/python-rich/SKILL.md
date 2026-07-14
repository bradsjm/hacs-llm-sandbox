---
name: python-rich
description: Build, refactor, debug, and test Python terminal output with Rich 14.x. Use when code imports `rich`, or when a task needs styled console output, tables, panels, trees, Markdown or syntax rendering, progress bars, spinners, live displays, layouts, prompts, rich logging or tracebacks, custom highlighters, pretty representations, or custom Rich renderables.
---

# Python Rich

Use Rich's highest-level renderable that fits the task. Preserve readable plain-text behavior when output is redirected or terminal capabilities are limited.

## Workflow

1. Inspect the project's Python version, dependency management, CLI framework, logging setup, output streams, tests, and existing `Console` ownership.
2. Confirm the installed Rich version when exact signatures matter. This skill is grounded in Rich 14.1.0; prefer the installed package or current upstream documentation when versions differ.
3. Select the smallest fitting API from the table below.
4. Reuse one application-level `Console` where practical, and pass it to components that must share output, themes, recording, or live-display state.
5. Separate data/work from presentation. Build renderables from values, then print them at the application boundary.
6. Verify normal terminal output, narrow-width behavior, redirected/plain-text output, and any interactive lifecycle changed by the task.

## API Selection

| Need | Prefer |
| --- | --- |
| Styled text or objects | `Console.print` |
| Structured text assembly | `Text` |
| Semantic reusable styles | `Theme` |
| Rows and columns | `Table` |
| Border around content | `Panel` |
| Hierarchy | `Tree` |
| Markdown or source code | `Markdown` or `Syntax` |
| Simple iterable progress | `track` |
| Multiple or customized tasks | `Progress` |
| Short indeterminate operation | `Console.status` |
| Arbitrary updating renderable | `Live` |
| Full-screen regions | `Layout` inside `Live(screen=True)` |
| Validated terminal input | `Prompt`, `IntPrompt`, `FloatPrompt`, or `Confirm` |
| Standard-library logging | `RichHandler` |
| Uncaught exception formatting | `rich.traceback.install` |
| Custom object presentation | `__rich__` or `__rich_console__` |

## Design Rules

- Prefer `Console.print` over shadowing built-in `print` in application modules. Use `rich.print` for small scripts or REPL work.
- Let Rich auto-detect terminal and color capabilities unless requirements explicitly demand an override.
- Keep stdout for primary output and use `Console(stderr=True)` for diagnostics or errors when stream separation matters.
- Treat markup as syntax. Use `markup=False` for literal strings, or `rich.markup.escape` for untrusted values interpolated into trusted markup.
- Prefer semantic theme names over repeated style strings when styles recur.
- Use context managers for `status`, `Progress`, `Live`, alternate screens, capture, and paging so cleanup occurs on exceptions.
- Print through `progress.console` or `live.console` while a dynamic display is active.
- Avoid `Segment`, manual ANSI codes, and low-level render methods unless implementing a custom renderable that cannot be composed from existing Rich objects.
- Keep animations modest. Lower refresh rates when work updates infrequently; use `auto_refresh=False` and explicit refreshes when deterministic control is needed.
- Do not force color or interactivity for pipes by default. For CI that supports ANSI but has no interactive user, use `TTY_COMPATIBLE=1` and `TTY_INTERACTIVE=0` when the deployment environment requires it.

## Testing

- Render to `Console(file=StringIO(), color_system=None, width=<fixed>)` for deterministic plain-text assertions.
- Assert meaningful content and layout behavior, not incidental ANSI sequences or every space unless exact formatting is the contract.
- Exercise narrow widths for tables, panels, wrapping, overflow, and custom renderables.
- Inject a shared console into progress, live, logging, or prompt components when output must be captured.
- Disable or manually refresh animation in tests rather than sleeping on timing-sensitive output.

## References

Load only the reference needed for the task:

| Reference | Use for |
| --- | --- |
| [core-console.md](references/core-console.md) | Console ownership, printing, styles, markup, capture/export, logging, tracebacks, and terminal detection |
| [static-renderables.md](references/static-renderables.md) | Text, tables, panels, trees, columns, groups, Markdown, syntax, padding, alignment, and layout composition |
| [dynamic-and-input.md](references/dynamic-and-input.md) | Status, progress, live displays, alternate screens, prompts, and lifecycle rules |
| [extension-protocols.md](references/extension-protocols.md) | Pretty reprs, custom highlighters, `__rich__`, `__rich_console__`, measurement, and low-level segments |
