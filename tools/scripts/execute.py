"""Execute command for APXM CLI."""

import subprocess
from pathlib import Path
from typing import Optional

import typer

from apxm_env import setup_mlir_environment
from apxm_styles import print_error, print_header, print_info

from tools.scripts import ensure_conda_env, get_config


def register_commands(app: typer.Typer) -> None:
    """Register execute commands on the app."""

    @app.command(context_settings={"allow_interspersed_args": False})
    def execute(
        file: Path = typer.Argument(..., help="ApxmGraph file to compile and execute"),
        args: Optional[list[str]] = typer.Argument(None, help="Arguments for entry flow"),
        opt_level: int = typer.Option(1, "-O", "--opt-level", help="Optimization level (0-3)"),
        trace: Optional[str] = typer.Option(
            None, "--trace", help="Enable tracing (levels: trace, debug, info, warn, error)"
        ),
        emit_metrics: Optional[Path] = typer.Option(
            None, "--emit-metrics", help="Emit runtime metrics to JSON file"
        ),
        cargo: bool = typer.Option(
            False,
            "--cargo",
            help="Use cargo run instead of pre-built binary (slower, but auto-rebuilds)",
        ),
    ):
        """Compile and execute an ApxmGraph file.

        Options must come BEFORE the file path. Arguments after file are passed to entry flow:
            apxm execute [options] workflow.json [args...]

        Examples:
            apxm execute workflow.json "quantum computing"
            apxm execute --emit-metrics metrics.json workflow.json "topic"
            apxm execute -O2 --trace debug workflow.json "input"
        """
        config = get_config()
        conda_prefix = ensure_conda_env()
        env = setup_mlir_environment(conda_prefix, config.target_dir)

        cwd = Path.cwd()
        file = file.resolve()
        if emit_metrics and not emit_metrics.is_absolute():
            emit_metrics = (cwd / emit_metrics).resolve()

        if not file.exists():
            print_error(f"File not found: {file}")
            raise typer.Exit(1)

        print_header(f"Executing {file.name}")

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
                "execute",
                f"-O{opt_level}",
            ]
            if trace:
                cmd.extend(["--trace", trace])
            if emit_metrics:
                cmd.extend(["--emit-metrics", str(emit_metrics)])
            cmd.append(str(file))
            if args:
                cmd.extend(args)
        else:
            if not config.compiler_bin.exists():
                print_error("Compiler not built!")
                print_info("Run: apxm build")
                print_info("Or use --cargo to auto-build")
                raise typer.Exit(1)
            cmd = [str(config.compiler_bin), "execute", f"-O{opt_level}"]
            if trace:
                cmd.extend(["--trace", trace])
            if emit_metrics:
                cmd.extend(["--emit-metrics", str(emit_metrics)])
            cmd.append(str(file))
            if args:
                cmd.extend(args)

        result = subprocess.run(cmd, cwd=config.apxm_dir, env=env)
        raise typer.Exit(result.returncode)
