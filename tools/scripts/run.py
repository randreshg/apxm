"""Run command for APXM CLI."""

import subprocess
from pathlib import Path
from typing import Optional

import typer

from apxm_env import setup_mlir_environment
from apxm_styles import print_error, print_header, print_info

from tools.scripts import ensure_conda_env, get_config


def register_commands(app: typer.Typer) -> None:
    """Register run commands on the app."""

    @app.command(context_settings={"allow_interspersed_args": False})
    def run(
        file: Path = typer.Argument(..., help="Compiled artifact (.apxmobj) to run"),
        args: Optional[list[str]] = typer.Argument(None, help="Arguments for entry flow"),
        trace: Optional[str] = typer.Option(
            None, "--trace", help="Enable tracing (levels: trace, debug, info, warn, error)"
        ),
        emit_metrics: Optional[Path] = typer.Option(
            None, "--emit-metrics", help="Emit runtime metrics to JSON file"
        ),
    ):
        """Run a pre-compiled artifact (.apxmobj).

        Use 'apxm execute <file.json>' to compile and run graph source files.
        """
        config = get_config()
        conda_prefix = ensure_conda_env()
        env = setup_mlir_environment(conda_prefix, config.target_dir)

        cwd = Path.cwd()
        file = file.resolve()
        if emit_metrics:
            emit_metrics = (
                (cwd / emit_metrics).resolve() if not emit_metrics.is_absolute() else emit_metrics
            )

        if not file.exists():
            print_error(f"File not found: {file}")
            raise typer.Exit(1)

        if file.suffix != ".apxmobj":
            print_error(f"Expected .apxmobj artifact file, got: {file.suffix or 'no extension'}")
            print_info("Use 'apxm execute <file.json>' to compile and run graph source files.")
            raise typer.Exit(1)

        if not config.compiler_bin.exists():
            print_error("Runtime not built!")
            print_info("Run: apxm build")
            raise typer.Exit(1)

        print_header(f"Running {file.name}")

        cmd = [str(config.compiler_bin), "run", str(file)]
        if trace:
            cmd.extend(["--trace", trace])
        if emit_metrics:
            cmd.extend(["--emit-metrics", str(emit_metrics)])
        if args:
            cmd.extend(args)
        result = subprocess.run(cmd, cwd=config.apxm_dir, env=env)
        raise typer.Exit(result.returncode)
