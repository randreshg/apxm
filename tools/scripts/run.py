"""Run command for APXM CLI."""

from pathlib import Path
from typing import Optional

from sniff import Typer, Argument, Option, Exit
from sniff import print_error, print_header, print_info

from . import get_config, ensure_binary, resolve_path, run_apxm


def register_commands(app: Typer) -> None:
    """Register run commands on the app."""

    @app.command(context_settings={"allow_interspersed_args": False})
    def run(
        file: Path = Argument(..., help="Compiled artifact (.apxmobj) to run"),
        args: Optional[list[str]] = Argument(None, help="Arguments for entry flow"),
        trace: Optional[str] = Option(
            None, "--trace", help="Enable tracing (levels: trace, debug, info, warn, error)"
        ),
        emit_metrics: Optional[Path] = Option(
            None, "--emit-metrics", help="Emit runtime metrics to JSON file"
        ),
    ):
        """Run a pre-compiled artifact (.apxmobj).

        Use 'apxm execute <file.json>' to compile and run graph source files.
        """
        config = get_config()
        cwd = Path.cwd()
        file = file.resolve()
        emit_metrics = resolve_path(emit_metrics, cwd)

        if not file.exists():
            print_error(f"File not found: {file}")
            raise Exit(1)

        if file.suffix != ".apxmobj":
            print_error(f"Expected .apxmobj artifact file, got: {file.suffix or 'no extension'}")
            print_info("Use 'apxm execute <file.json>' to compile and run graph source files.")
            raise Exit(1)

        ensure_binary(config)
        print_header(f"Running {file.name}")

        cmd = [str(config.compiler_bin), "run", str(file)]
        if trace:
            cmd.extend(["--trace", trace])
        if emit_metrics:
            cmd.extend(["--emit-metrics", str(emit_metrics)])
        if args:
            cmd.extend(args)

        raise Exit(run_apxm(config, cmd))
