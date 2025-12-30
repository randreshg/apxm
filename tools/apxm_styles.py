#!/usr/bin/env python3
"""
APXM Shared Styles - Common Rich console styling for APXM CLI tools.
"""

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Global Console
console = Console()


class Colors:
    """APXM color scheme for consistent styling."""
    SUCCESS = "bold green"
    ERROR = "bold red"
    WARNING = "bold yellow"
    INFO = "cyan"
    DEBUG = "dim"
    HEADER = "bold cyan"
    STEP = "bold blue"
    DIM = "dim"
    HIGHLIGHT = "bold white"
    PASS = "green"
    FAIL = "red"
    SKIP = "yellow"
    PENDING = "dim"
    RUNNING = "blue"


class Symbols:
    """Unicode symbols for status indicators."""
    PASS = "\u2713"      # checkmark
    FAIL = "\u2717"      # x
    SKIP = "\u25cb"      # circle
    TIMEOUT = "\u23f1"   # stopwatch
    RUNNING = "\u25cf"   # filled circle
    INFO = "\u2139"      # i
    WARNING = "\u26a0"   # warning


def print_header(title: str, subtitle: str | None = None) -> None:
    """Print a styled header panel with optional subtitle."""
    console.print()
    header_text = Text(title, style=Colors.HEADER)
    if subtitle:
        header_text.append("\n", style=Colors.DIM)
        header_text.append(subtitle, style=Colors.DIM)
    console.print(
        Panel(
            header_text,
            box=box.HEAVY,
            border_style="cyan",
            padding=(0, 1),
        )
    )
    console.print()


def print_footer(title: str, style: str = "green") -> None:
    """Print a styled footer panel."""
    console.print()
    console.print(
        Panel(
            Text(title, style=style),
            box=box.HEAVY,
            border_style=style,
            padding=(0, 1),
        )
    )
    console.print()


def print_step(msg: str, step_num: int | None = None, total: int | None = None) -> None:
    """Print a step indicator with enhanced formatting."""
    if step_num is not None and total is not None:
        prefix = f"[{Colors.STEP}][{step_num}/{total}][/{Colors.STEP}]"
        arrow = f"[{Colors.STEP}]\u25b6[/{Colors.STEP}]"
        console.print(f"  {prefix} {arrow} {msg}")
    else:
        arrow = f"[{Colors.STEP}]\u25b6[/{Colors.STEP}]"
        console.print(f"  {arrow} {msg}")


def print_success(msg: str) -> None:
    """Print a success message with icon."""
    icon = f"[{Colors.SUCCESS}]{Symbols.PASS}[/{Colors.SUCCESS}]"
    console.print(f"  {icon} [{Colors.SUCCESS}]{msg}[/{Colors.SUCCESS}]")


def print_error(msg: str) -> None:
    """Print an error message with icon."""
    icon = f"[{Colors.ERROR}]{Symbols.FAIL}[/{Colors.ERROR}]"
    console.print(f"  {icon} [{Colors.ERROR}]{msg}[/{Colors.ERROR}]")


def print_warning(msg: str) -> None:
    """Print a warning message with icon."""
    icon = f"[{Colors.WARNING}]{Symbols.WARNING}[/{Colors.WARNING}]"
    console.print(f"  {icon} [{Colors.WARNING}]{msg}[/{Colors.WARNING}]")


def print_info(msg: str) -> None:
    """Print an info message with icon."""
    icon = f"[{Colors.INFO}]{Symbols.INFO}[/{Colors.INFO}]"
    console.print(f"  {icon} [{Colors.INFO}]{msg}[/{Colors.INFO}]")


def print_debug(msg: str) -> None:
    """Print a debug message with icon."""
    icon = f"[{Colors.DEBUG}]\u25cf[/{Colors.DEBUG}]"
    console.print(f"  {icon} [{Colors.DEBUG}]{msg}[/{Colors.DEBUG}]", style=Colors.DEBUG)


VERSION = "0.1.0"
