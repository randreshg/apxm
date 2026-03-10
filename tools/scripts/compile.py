"""Compile command for APXM CLI."""

import subprocess
from pathlib import Path
from typing import Optional

import typer

from apxm_env import setup_mlir_environment
from apxm_styles import print_error, print_info, print_step, print_success

from tools.scripts import ensure_conda_env, get_config


def register_commands(app: typer.Typer) -> None:
    """Register compile commands on the app."""

    @app.command()
    def compile(
        file: Path = typer.Argument(..., help="ApxmGraph file to compile"),
        output: Path = typer.Option(..., "-o", "--output", help="Output artifact path"),
        emit_diagnostics: Optional[Path] = typer.Option(
            None, "--emit-diagnostics", help="Emit diagnostics JSON file"
        ),
        opt_level: int = typer.Option(1, "-O", "--opt-level", help="Optimization level (0-3)"),
        cargo: bool = typer.Option(
            False,
            "--cargo",
            help="Use cargo run instead of pre-built binary (slower, but auto-rebuilds)",
        ),
    ):
        """Compile an ApxmGraph file to an artifact (.apxmobj)."""
        config = get_config()
        conda_prefix = ensure_conda_env()
        env = setup_mlir_environment(conda_prefix, config.target_dir)

        cwd = Path.cwd()
        file = file.resolve()
        output = (cwd / output).resolve() if not output.is_absolute() else output
        if emit_diagnostics:
            emit_diagnostics = (
                (cwd / emit_diagnostics).resolve()
                if not emit_diagnostics.is_absolute()
                else emit_diagnostics
            )

        if not file.exists():
            print_error(f"File not found: {file}")
            raise typer.Exit(1)

        print_step(f"Compiling {file.name}...")

        if cargo:
            cmd = [
                "cargo",
                "run",
                "-p",
                "apxm-cli",
                "--features",
                "driver,metrics",
                "--release",
                "--",
                "compile",
                str(file),
                "-o",
                str(output),
                f"-O{opt_level}",
            ]
            if emit_diagnostics:
                cmd.extend(["--emit-diagnostics", str(emit_diagnostics)])
        else:
            if not config.compiler_bin.exists():
                print_error("Compiler not built!")
                print_info("Run: apxm build")
                print_info("Or use --cargo to auto-build")
                raise typer.Exit(1)
            cmd = [
                str(config.compiler_bin),
                "compile",
                str(file),
                "-o",
                str(output),
                f"-O{opt_level}",
            ]
            if emit_diagnostics:
                cmd.extend(["--emit-diagnostics", str(emit_diagnostics)])

        result = subprocess.run(cmd, cwd=config.apxm_dir, env=env)

        if result.returncode == 0:
            print_success(f"Compiled to {output}")
        else:
            print_error("Compilation failed!")

        raise typer.Exit(result.returncode)
