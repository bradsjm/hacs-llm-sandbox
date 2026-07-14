# Dynamic Displays and Input

This reference targets Rich 14.1.0.

## Contents

- Status indicators
- Progress displays
- Live displays
- Alternate screen and nesting
- Prompts
- Testing dynamic output

## Status Indicators

Use a status context for one indeterminate operation:

```python
with console.status("Loading...", spinner="dots"):
    load_data()
```

The context manager starts and stops the spinner. Use the returned `Status` object's `update` method only when the message or spinner must change during the operation.

## Progress Displays

Use `track` for one iterable:

```python
from rich.progress import track

for item in track(items, description="Processing..."):
    process(item)
```

Supply `total` for iterables without `len`. Use `Progress` for multiple tasks, custom columns, manually reported completion, or shared console output:

```python
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

with Progress(
    TextColumn("{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    console=console,
) as progress:
    task_id = progress.add_task("Downloading", total=byte_count)
    for chunk in chunks:
        write(chunk)
        progress.advance(task_id, len(chunk))
```

Important behavior:

- Use `total=None` or `start=False` for indeterminate work. Call `start_task` when bounded work begins.
- Use `update(task_id, completed=...)` for absolute values or `advance=...` / `advance(...)` for deltas.
- Store custom display values through task fields and reference them as `{task.fields[name]}` in `TextColumn`.
- Set `visible=False` to hide tasks without deleting them.
- Set `transient=True` to clear the display on exit.
- Lower `refresh_per_second` when updates are infrequent.
- Use `auto_refresh=False` and call `refresh` for deterministic or event-driven updates.
- Print messages through `progress.console` so they appear above the progress display.
- Use `rich.progress.open` or `wrap_file` for byte-level file reading progress.

Available columns include `SpinnerColumn`, `BarColumn`, `TaskProgressColumn`, `TextColumn`, `TimeElapsedColumn`, `TimeRemainingColumn`, `MofNCompleteColumn`, `DownloadColumn`, `FileSizeColumn`, `TotalFileSizeColumn`, and `TransferSpeedColumn`.

One `Progress` has one column definition for all tasks. Combine multiple `Progress` renderables in a `Live` display when task groups need different columns.

## Live Displays

Use `Live` for any renderable that changes over time:

```python
from rich.live import Live

with Live(build_table(), console=console, refresh_per_second=4) as live:
    for event in events:
        apply(event)
        live.update(build_table())
```

Mutate the current renderable in place when that is simpler, or replace it with `live.update`. Pass `refresh=True` when auto-refresh is disabled or an immediate update is required.

Use `vertical_overflow="ellipsis"`, `"crop"`, or `"visible"` based on whether unseen rows should be signaled, hidden, or allowed to scroll. `visible` cannot be cleanly erased while live rendering.

Print through `live.console` during a live display. Rich redirects stdout and stderr by default; disable `redirect_stdout` or `redirect_stderr` only when the application owns those streams separately.

## Alternate Screen and Nesting

Use `Live(screen=True)` for a full-screen interface that restores the regular terminal on exit. For simpler cases, use `console.screen()` as a context manager. Do not call `set_alt_screen` directly unless manual lifecycle control is required.

Rich 14 supports nested `Live` and nested progress displays. Inner content appears below outer content and refreshes according to the outer display's lifecycle. Keep nesting shallow and share one console.

## Prompts

Use the typed prompt matching the result:

```python
from rich.prompt import Confirm, IntPrompt, Prompt

name = Prompt.ask("Name", default="Ada")
retries = IntPrompt.ask("Retries", default=3)
proceed = Confirm.ask("Continue?", default=True)
```

Pass `choices=[...]` for enumerated input and `case_sensitive=False` when case should not matter. Use `password=True` for hidden terminal entry. Subclass `PromptBase` and override `process_response` only when built-in typed prompts and choices cannot express validation.

Avoid prompting in non-interactive commands unless the command contract explicitly allows it. Provide flags or injected input streams for automation and tests.

## Testing Dynamic Output

- Inject a fixed-width `Console(file=StringIO(), color_system=None)`.
- Set `auto_refresh=False` and call `refresh` at known points.
- Use `disable=True` when testing work logic independently of progress presentation.
- Avoid sleep-based assertions and frame-by-frame animation snapshots.
- Assert final content, task state, and cleanup behavior rather than timing-dependent cursor control sequences.
