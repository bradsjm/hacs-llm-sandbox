# Static Renderables

This reference targets Rich 14.1.0.

## Contents

- Composition principles
- Text and alignment
- Tables and grids
- Panels, padding, columns, and groups
- Trees
- Markdown and syntax
- Layout

## Composition Principles

Rich renderables compose: a table cell may contain a `Text`, `Panel`, `Tree`, `Syntax`, or another table. Prefer composition over custom rendering. Build a renderable in a function and let the caller print it.

## Text and Alignment

Use `Text` when styling regions or manipulating styled content:

```python
from rich.text import Text

message = Text.assemble(
    ("PASS", "bold green"),
    " 42 checks",
)
```

`Text` is mutable. Use `append`, `stylize`, `highlight_words`, or `highlight_regex` for incremental changes. Use `Text.from_markup` for trusted markup and `Text.from_ansi` when converting existing ANSI output.

Control wrapping with `justify`, `overflow`, and `no_wrap`. Wrap a renderable in `Align` for horizontal or vertical placement. Use `Padding(renderable, 1)` or CSS-like tuples `(vertical, horizontal)` and `(top, right, bottom, left)` for spacing.

## Tables and Grids

Construct a `Table`, define columns, add rows, then print it:

```python
from rich.table import Table

table = Table(title="Jobs", row_styles=["", "dim"])
table.add_column("ID", justify="right", no_wrap=True)
table.add_column("State")
table.add_row("17", "[green]complete[/green]")
```

Key table controls:

- `box`, `border_style`, `show_edge`, `show_lines`: border behavior.
- `expand`, `width`, `min_width`: overall sizing.
- `padding`, `collapse_padding`, `pad_edge`: cell spacing.
- `row_styles`: alternating row styles.
- `safe_box`: restrict problematic box characters on legacy terminals.

Key column controls:

- `justify` and `vertical`: cell alignment.
- `width`, `min_width`, `max_width`: fixed or bounded sizing.
- `ratio`: flexible width when table width is allocated.
- `no_wrap` and `overflow`: narrow-terminal behavior.

Use `Table.grid()` for borderless layout. It is suitable for concise key/value lines and left/right alignment, not semantic tabular data that should retain headers.

## Panels, Columns, and Groups

Use `Panel(content, title=...)` for a full-width border. Use `Panel.fit(...)` or `expand=False` when the panel should fit its content.

Use `Columns(renderables, equal=True, expand=True)` for wrapping collections of cards or labels across the available width.

Use `Group(*renderables)` when an API accepts one renderable but the UI requires several stacked items. Use the `@group()` generator decorator for dynamic groups.

## Trees

Use `Tree` for hierarchical data:

```python
from rich.tree import Tree

root = Tree("project")
source = root.add("src", guide_style="dim")
source.add("app.py")
source.add("models.py")
```

`add` returns the child `Tree`, enabling nested construction. Use `style` for branch content and `guide_style` for connector lines. Use `hide_root=True` when only children should appear.

## Markdown and Syntax

Render Markdown with `Markdown(markup, code_theme=..., hyperlinks=True)`. Use it for trusted documentation or help content. Code blocks receive syntax highlighting.

Render source with `Syntax(code, lexer, ...)` or `Syntax.from_path(path, ...)`. Useful options include:

- `line_numbers=True`
- `highlight_lines={...}`
- `line_range=(start, end)`
- `word_wrap=True`
- `indent_guides=True`
- `background_color="default"`

Prefer `Syntax.from_path` when the source comes from a file because it can infer the lexer.

## Layout

Use `Layout` to divide a known console region. Split vertically with `split_column` and horizontally with `split_row`:

```python
from rich.layout import Layout

layout = Layout(name="root")
layout.split_column(
    Layout(name="header", size=3),
    Layout(name="body", ratio=1, minimum_size=10),
)
layout["body"].split_row(Layout(name="main", ratio=2), Layout(name="side"))
```

Set content with `layout["name"].update(renderable)`. Use `size` for fixed cells, `ratio` for flexible allocation, `minimum_size` to prevent collapse, and `visible` to toggle regions. Print `layout.tree` while debugging structure.

Use `Layout` inside `Live`, often with `screen=True`, for full-screen terminal interfaces. Do not use it when a table or grid sufficiently expresses the output.
