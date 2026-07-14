# Core Console

This reference targets Rich 14.1.0.

## Contents

- Console ownership
- Printing and streams
- Styles, themes, and markup
- Capture and export
- Logging and tracebacks
- Terminal behavior

## Console Ownership

Create one shared `Console` for most applications:

```python
from rich.console import Console

console = Console()
```

Pass that instance to `Progress`, `Live`, `Status`, `RichHandler`, and prompts when they must coordinate output. Rich also exposes a global console through `rich.get_console()`, but explicit ownership is easier to configure and test.

Useful constructor controls include `stderr`, `file`, `width`, `height`, `theme`, `record`, `markup`, `highlight`, `color_system`, `force_terminal`, and `force_interactive`.

## Printing and Streams

Use `console.print` for strings, containers, and renderables. Important controls include:

- `style`: apply a style to the whole output.
- `justify`: `default`, `left`, `center`, `right`, or `full`.
- `overflow`: `fold`, `crop`, `ellipsis`, or `ignore`.
- `markup` and `highlight`: override console defaults per call.
- `soft_wrap=True`: disable Rich word wrapping and cropping.
- `crop=False`: allow content beyond the console width when appropriate.

Use `console.log` for developer-oriented output with time and call-site columns. Set `log_locals=True` only when exposing local values is acceptable.

Use `console.print_json(json=...)` for encoded JSON or `console.print_json(data=...)` for Python data. Use `console.rule(title)` to divide output and `console.input` only for unvalidated raw input.

Create a separate error stream when needed:

```python
error_console = Console(stderr=True, style="bold red")
```

## Styles, Themes, and Markup

Style strings combine foreground, background, and attributes:

```python
console.print("Warning", style="bold yellow on red")
console.print("Custom", style="#af00ff")
```

Use `Style` when constructing or combining styles programmatically. Use `Theme` for repeated semantic names:

```python
from rich.console import Console
from rich.theme import Theme

theme = Theme({
    "info": "dim cyan",
    "warning": "bold yellow",
    "danger": "bold red",
})
console = Console(theme=theme)
console.print("Operation failed", style="danger")
```

Theme names must be lowercase, start with a letter, and contain only letters, `.`, `-`, or `_`.

Markup uses tags such as `[bold red]text[/]` and `[link=https://example.com]site[/link]`. Escape dynamic values before inserting them into trusted markup:

```python
from rich.markup import escape

console.print(f"Hello [bold]{escape(user_name)}[/bold]")
```

If the entire string is literal data, prefer `console.print(value, markup=False)`.

## Capture and Export

For ad hoc capture:

```python
with console.capture() as capture:
    console.print("[bold]Hello[/]")
output = capture.get()
```

For tests, prefer an injected `StringIO`:

```python
from io import StringIO
from rich.console import Console

stream = StringIO()
console = Console(file=stream, color_system=None, width=80)
console.print("[bold]Hello[/]")
assert "Hello" in stream.getvalue()
```

Construct with `record=True` before calling `export_text`, `export_html`, `export_svg`, or their `save_*` counterparts. Export methods clear the recording by default; pass `clear=False` when reusing it.

## Logging and Tracebacks

Use `RichHandler` with standard-library logging:

```python
import logging
from rich.logging import RichHandler

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
```

Logging markup defaults to disabled. Keep it disabled for messages from libraries or uncontrolled sources. Enable it globally with `markup=True` or per record with `extra={"markup": True}` only for trusted messages.

For handled exceptions, call `console.print_exception(show_locals=True)`. For uncaught exceptions, install the handler at the application entry point:

```python
from rich.traceback import install

install(show_locals=True, max_frames=100)
```

Use `suppress=[module_or_path]` to collapse framework frames. Avoid exposing locals when they may contain secrets or large objects.

## Terminal Behavior

Rich strips control codes when output is not a terminal and removes animations in non-interactive output. Prefer this auto-detection.

- `NO_COLOR`: remove color while preserving attributes such as bold.
- `FORCE_COLOR`: enable color unless `NO_COLOR` is set.
- `TTY_COMPATIBLE=1`: treat output as ANSI-capable.
- `TTY_INTERACTIVE=0`: disable interactive animation.
- `COLUMNS` and `LINES`: provide dimensions when constructor dimensions are absent.

Set `force_terminal` or `force_interactive` only when auto-detection is wrong for a known environment. A higher forced color system than the terminal supports can make output unreadable.
