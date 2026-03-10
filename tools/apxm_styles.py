#!/usr/bin/env python3
"""
APXM Shared Styles - Common Rich console styling for all APXM CLI tools.

This module provides consistent styling, colors, and output helpers
across the apxm CLI, examples, and test runners.
"""

from enum import Enum
from typing import Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

# ============================================================================
# Global Console
# ============================================================================

console = Console()

# ============================================================================
# Color Scheme
# ============================================================================


class Colors:
    """APXM color scheme for consistent styling."""
    # Status colors
    SUCCESS = "bold green"
    ERROR = "bold red"
    WARNING = "bold yellow"
    INFO = "cyan"
    DEBUG = "dim"

    # UI colors
    HEADER = "bold cyan"
    STEP = "bold blue"
    DIM = "dim"
    HIGHLIGHT = "bold white"

    # Status-specific
    PASS = "green"
    FAIL = "red"
    SKIP = "yellow"
    PENDING = "dim"
    RUNNING = "blue"


# ============================================================================
# Status Enum
# ============================================================================


class Status(str, Enum):
    """Status of a build, test, or run operation."""
    PENDING = "pending"
    BUILDING = "building"
    RUNNING = "running"
    PASS = "pass"
    FAIL = "fail"
    CRASH = "crash"
    TIMEOUT = "timeout"
    SKIP = "skip"

    def style(self) -> str:
        """Get the Rich style for this status."""
        styles = {
            Status.PENDING: Colors.PENDING,
            Status.BUILDING: Colors.RUNNING,
            Status.RUNNING: Colors.RUNNING,
            Status.PASS: Colors.PASS,
            Status.FAIL: Colors.FAIL,
            Status.CRASH: Colors.FAIL,
            Status.TIMEOUT: Colors.WARNING,
            Status.SKIP: Colors.SKIP,
        }
        return styles.get(self, Colors.DIM)

    def symbol(self) -> str:
        """Get a display symbol for this status."""
        symbols = {
            Status.PENDING: "...",
            Status.BUILDING: "...",
            Status.RUNNING: "...",
            Status.PASS: "OK",
            Status.FAIL: "FAIL",
            Status.CRASH: "CRASH",
            Status.TIMEOUT: "TIMEOUT",
            Status.SKIP: "SKIP",
        }
        return symbols.get(self, "?")


# ============================================================================
# Symbols
# ============================================================================


class Symbols:
    """Unicode symbols for status indicators."""
    PASS = "\u2713"      # checkmark
    FAIL = "\u2717"      # x
    SKIP = "\u25cb"      # circle
    TIMEOUT = "\u23f1"   # stopwatch
    RUNNING = "\u25cf"   # filled circle
    INFO = "\u2139"      # i
    WARNING = "\u26a0"   # warning


# ============================================================================
# Output Helpers
# ============================================================================


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
            Text.from_markup(title, style=style),
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


def print_status(name: str, status: Status, note: str = "") -> None:
    """Print a status line for an item (dotted leader style)."""
    status_text = f"[{status.style()}]{status.symbol()}[/{status.style()}]"
    if note:
        console.print(f"  {name:.<40} {status_text} {note}")
    else:
        console.print(f"  {name:.<40} {status_text}")


# ============================================================================
# Status Formatting Helpers
# ============================================================================


def format_passed(count: int) -> str:
    """Format passed count with color and symbol."""
    return f"[{Colors.PASS}]{Symbols.PASS} {count}[/{Colors.PASS}] passed"


def format_failed(count: int) -> str:
    """Format failed count with color and symbol."""
    return f"[{Colors.FAIL}]{Symbols.FAIL} {count}[/{Colors.FAIL}] failed"


def format_skipped(count: int) -> str:
    """Format skipped count with color and symbol."""
    return f"[{Colors.DIM}]{Symbols.SKIP} {count}[/{Colors.DIM}] skipped"


def format_summary_line(passed: int, failed: int, skipped: int) -> str:
    """Format a complete summary line with all counts."""
    return f"{format_passed(passed)}  {format_failed(failed)}  {format_skipped(skipped)}"


def status_symbol(status_value: str) -> str:
    """Get a colored symbol for a status string."""
    if status_value in ("PASS", "pass"):
        return f"[{Colors.PASS}]{Symbols.PASS}[/{Colors.PASS}]"
    elif status_value in ("FAIL", "fail", "CRASH", "crash"):
        return f"[{Colors.FAIL}]{Symbols.FAIL}[/{Colors.FAIL}]"
    elif status_value in ("SKIP", "skip"):
        return f"[{Colors.DIM}]{Symbols.SKIP}[/{Colors.DIM}]"
    elif status_value in ("TIMEOUT", "timeout"):
        return f"[{Colors.WARNING}]{Symbols.TIMEOUT}[/{Colors.WARNING}]"
    else:
        return f"[{Colors.DIM}]-[/{Colors.DIM}]"


# ============================================================================
# Table Helpers
# ============================================================================


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Print a formatted table using Rich."""
    table = Table(title=title, box=box.SIMPLE_HEAVY, border_style="cyan")
    for header in headers:
        table.add_column(header, style="bold")
    for row in rows:
        table.add_row(*row)
    console.print(table)


def create_results_table(title: str = "Results") -> Table:
    """Create a standard results table."""
    return Table(
        title=title,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
    )


def create_summary_panel(content: str, title: str = "Summary", style: str = "green") -> Panel:
    """Create a summary panel."""
    return Panel(content, title=title, border_style=style)


# ============================================================================
# Progress Helpers
# ============================================================================


def create_progress(description: str = "Processing...") -> Progress:
    """Create a standard progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def create_spinner_progress() -> Progress:
    """Create a spinner-only progress (no bar)."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )


# ============================================================================
# Section / Layout Helpers
# ============================================================================


def print_section(title: str) -> None:
    """Print a bold section label (e.g., 'Dependencies:')."""
    console.print(f"\n[bold]{title}[/bold]")


def print_dim(msg: str) -> None:
    """Print dim/muted text."""
    console.print(f"  [dim]{msg}[/dim]")


def print_numbered_list(items: list[str]) -> None:
    """Print a numbered list of action items."""
    for i, item in enumerate(items, 1):
        console.print(f"  [{Colors.WARNING}]{i}.[/{Colors.WARNING}] {item}")


def print_next_steps(steps: list[str]) -> None:
    """Print a 'Next steps' block."""
    console.print()
    console.print("[bold]Next steps:[/bold]")
    for step in steps:
        console.print(f"  {step}")


def print_blank() -> None:
    """Print a blank line for spacing."""
    console.print()


# ============================================================================
# Version
# ============================================================================

VERSION = "0.2.0"
