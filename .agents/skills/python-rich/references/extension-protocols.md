# Extension Protocols

This reference targets Rich 14.1.0.

## Contents

- Rich repr protocol
- Custom highlighters
- Simple custom renderables
- Generator renderables
- Measurement
- Low-level segments

## Rich Repr Protocol

Use `__rich_repr__` to improve pretty printing of custom objects without controlling their full visual layout:

```python
import rich.repr

class Job:
    def __init__(self, job_id: int, state: str, retries: int = 0) -> None:
        self.job_id = job_id
        self.state = state
        self.retries = retries

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.job_id
        yield "state", self.state
        yield "retries", self.retries, 0
```

Yield `value` for a positional argument, `(name, value)` for a keyword argument, and `(name, value, default)` to omit defaults. Set `__rich_repr__.angular = True` for angle-bracket representation. Use `@rich.repr.auto` only when constructor parameter names reliably match attributes.

## Custom Highlighters

Extend `RegexHighlighter` for named regular-expression matches:

```python
from rich.highlighter import RegexHighlighter
from rich.theme import Theme

class TicketHighlighter(RegexHighlighter):
    base_style = "ticket."
    highlights = [r"(?P<id>PROJ-\d+)"]

theme = Theme({"ticket.id": "bold cyan"})
console = Console(highlighter=TicketHighlighter(), theme=theme)
```

Named capture groups are appended to `base_style`. Extend `Highlighter` directly and implement `highlight(self, text: Text) -> None` only for behavior that regexes cannot express. Highlighters mutate `Text` in place.

## Simple Custom Renderables

Implement `__rich__` when an object can return one existing renderable:

```python
from rich.text import Text

class Health:
    def __rich__(self) -> Text:
        return Text("healthy", style="bold green")
```

A returned string is interpreted as console markup, so return `Text` when content may contain literal brackets or when style boundaries should be explicit.

## Generator Renderables

Implement `__rich_console__` when output requires multiple renderables or depends on console options:

```python
from rich.console import Console, ConsoleOptions, RenderResult
from rich.table import Table

class User:
    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        yield "[bold]User[/bold]"
        table = Table.grid()
        table.add_row("Name", self.name)
        table.add_row("Role", self.role)
        yield table
```

Treat `ConsoleOptions` as the rendering contract. Respect `max_width`, `max_height`, `ascii_only`, markup, wrapping, and overflow where the custom output needs them. Yield existing renderables whenever possible.

## Measurement

Implement `__rich_measure__` only when Rich cannot infer useful dimensions from yielded renderables:

```python
from rich.measure import Measurement

def __rich_measure__(self, console: Console, options: ConsoleOptions) -> Measurement:
    return Measurement(minimum=8, maximum=options.max_width)
```

Return the minimum and maximum terminal-cell widths required, not Python string lengths. Wide Unicode characters may occupy more than one cell.

## Low-Level Segments

Yield `Segment(text, style)` only when composition from `Text`, `Table`, `Panel`, or other standard renderables cannot express the output. Segments are terminal-cell-oriented and may carry control codes.

If segment-level work is unavoidable:

- Use Rich's `Style`, not handwritten ANSI escapes.
- Preserve newlines deliberately.
- Account for cell width rather than code-point count.
- Test ASCII-only and narrow-width rendering.
- Add `__rich_measure__` when the object participates in table or layout sizing.

Use `rich.protocol.is_renderable` to check support and `rich.protocol.rich_cast` to invoke `__rich__` recursively when building library-level integrations.
