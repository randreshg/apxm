"""Compile command for APXM CLI."""

from pathlib import Path
from typing import Optional

from sniff import Typer, Argument, Option, Exit
from sniff import print_error, print_step, print_success

from . import get_config, build_apxm_cmd, resolve_path, run_apxm


def register_commands(app: Typer) -> None:
    """Register compile commands on the app."""

    @app.command()
    def compile(
        file: Path = Argument(..., help="ApxmGraph file to compile"),
        output: Path = Option(..., "-o", "--output", help="Output artifact path"),
        emit_diagnostics: Optional[Path] = Option(
            None, "--emit-diagnostics", help="Emit diagnostics JSON file"
        ),
        opt_level: int = Option(1, "-O", "--opt-level", help="Optimization level (0-3)"),
        cargo: bool = Option(
            False,
            "--cargo",
            help="Use cargo run instead of pre-built binary (slower, but auto-rebuilds)",
        ),
    ):
        """Compile an ApxmGraph file to an artifact (.apxmobj)."""
        config = get_config()
        cwd = Path.cwd()
        file = file.resolve()
        output = resolve_path(output, cwd) or output
        emit_diagnostics = resolve_path(emit_diagnostics, cwd)

        if not file.exists():
            print_error(f"File not found: {file}")
            raise Exit(1)

        print_step(f"Compiling {file.name}...")

        extra = [str(file), "-o", str(output), f"-O{opt_level}"]
        if emit_diagnostics:
            extra.extend(["--emit-diagnostics", str(emit_diagnostics)])

        cmd = build_apxm_cmd(config, "compile", extra, cargo=cargo)
        rc = run_apxm(config, cmd)

        if rc == 0:
            print_success(f"Compiled to {output}")
        else:
            print_error("Compilation failed!")
        raise Exit(rc)
