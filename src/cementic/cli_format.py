"""Click/Typer subclasses giving cementic's CLI plain, case-consistent help text.

Split out of cli.py: these six classes are pure formatting machinery with no
dependency on any command, config, or database code.
"""

import inspect
from collections.abc import Callable
from typing import Any, TypeVar, cast

import click
import typer
from typer.core import TyperCommand, TyperGroup

_CommandFn = TypeVar("_CommandFn", bound=Callable[..., Any])


class _UpperFormatter(click.HelpFormatter):
    """Plain help formatter that uppercases section headings (USAGE, OPTIONS, …).

    Keeps the headings consistent with Click's uppercase usage metavars
    (``[OPTIONS] COMMAND [ARGS]``).
    """

    def write_heading(self, heading: str) -> None:
        super().write_heading(heading.upper())

    def write_usage(self, prog: str, args: str = "", prefix: str | None = None) -> None:
        super().write_usage(prog, args, prefix=(prefix or "Usage: ").upper())


class _UpperContext(click.Context):
    def make_formatter(self) -> click.HelpFormatter:
        return _UpperFormatter(width=self.terminal_width, max_width=self.max_content_width)


class _PlainEpilogMixin:
    """Render the epilog at the base indent so its own headers line up with
    USAGE/OPTIONS/COMMANDS (Click otherwise indents the whole epilog)."""

    epilog: str | None

    def format_epilog(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        if self.epilog:
            formatter.write_paragraph()
            formatter.write_text(inspect.cleandoc(self.epilog))


class _UpperGroup(_PlainEpilogMixin, TyperGroup):
    context_class = _UpperContext


class _UpperCommand(_PlainEpilogMixin, TyperCommand):
    context_class = _UpperContext


class CementicTyper(typer.Typer):
    """Typer app with plain, case-consistent help formatting."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("rich_markup_mode", None)
        kwargs.setdefault("add_completion", False)
        kwargs.setdefault("no_args_is_help", True)
        kwargs.setdefault("cls", _UpperGroup)
        context_settings = cast(dict[str, Any], dict(kwargs.get("context_settings") or {}))
        context_settings.setdefault("help_option_names", ["-h", "--help"])
        kwargs["context_settings"] = context_settings
        super().__init__(*args, **kwargs)

    def command(self, *args: Any, **kwargs: Any) -> Callable[[_CommandFn], _CommandFn]:
        kwargs.setdefault("cls", _UpperCommand)
        return super().command(*args, **kwargs)
