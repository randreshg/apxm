"""Execute command for APXM CLI."""

from pathlib import Path
from typing import Optional

from sniff import Typer, Argument, Option, Exit
from sniff import print_error, print_header

from . import get_config, build_apxm_cmd, resolve_path, run_apxm


def register_commands(app: Typer) -> None:
    """Register execute commands on the app."""

    @app.command(context_settings={"allow_interspersed_args": False})
    def execute(
        file: Path = Argument(..., help="ApxmGraph file to compile and execute"),
        args: Optional[list[str]] = Argument(None, help="Arguments for entry flow"),
        opt_level: int = Option(1, "-O", "--opt-level", help="Optimization level (0-3)"),
        trace: Optional[str] = Option(
            None, "--trace", help="Enable tracing (levels: trace, debug, info, warn, error)"
        ),
        emit_metrics: Optional[Path] = Option(
            None, "--emit-metrics", help="Emit runtime metrics to JSON file"
        ),
        cargo: bool = Option(
            False,
            "--cargo",
            help="Use cargo run instead of pre-built binary (slower, but auto-rebuilds)",
        ),
    ):
        """Compile and execute an ApxmGraph file.

        Options must come BEFORE the file path. Arguments after file are passed to entry flow:
            apxm execute [options] workflow.json [args...]
        """
        config = get_config()
        cwd = Path.cwd()
        file = file.resolve()
        emit_metrics = resolve_path(emit_metrics, cwd)

        if not file.exists():
            print_error(f"File not found: {file}")
            raise Exit(1)

        print_header(f"Executing {file.name}")

        extra = [f"-O{opt_level}"]
        if trace:
            extra.extend(["--trace", trace])
        if emit_metrics:
            extra.extend(["--emit-metrics", str(emit_metrics)])
        extra.append(str(file))
        if args:
            extra.extend(args)

        cmd = build_apxm_cmd(config, "execute", extra, cargo=cargo)
        raise Exit(run_apxm(config, cmd))
