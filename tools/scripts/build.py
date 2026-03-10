"""Build command for APXM CLI."""

import shutil
import subprocess

import typer

from apxm_env import setup_mlir_environment
from apxm_styles import print_error, print_header, print_info, print_step, print_success, print_warning

from tools.scripts import ensure_conda_env, get_config


def register_commands(app: typer.Typer) -> None:
    """Register build commands on the app."""

    @app.command()
    def build(
        compiler: bool = typer.Option(False, "--compiler", "-c", help="Build compiler only"),
        runtime: bool = typer.Option(False, "--runtime", "-r", help="Build runtime only"),
        release: bool = typer.Option(True, "--release/--debug", help="Build in release mode"),
        clean: bool = typer.Option(False, "--clean", help="Clean build artifacts first"),
        trace: bool = typer.Option(False, "--trace", help="Build with tracing enabled (default)"),
        no_trace: bool = typer.Option(
            False, "--no-trace", help="Build with tracing disabled (zero overhead)"
        ),
    ):
        """Build APXM components.

        By default, builds the full project (compiler + runtime) with tracing enabled.
        Use --compiler or --runtime to build specific components.

        Tracing control:
          --trace     Build with runtime tracing enabled (default)
          --no-trace  Build with tracing compiled out (zero overhead)

        When built with --trace (or default), use 'apxm run --trace <level>' to control
        output at runtime. When built with --no-trace, all tracing is eliminated at
        compile time for maximum performance.
        """
        config = get_config()
        conda_prefix = ensure_conda_env()
        env = setup_mlir_environment(conda_prefix, config.target_dir)

        if trace and no_trace:
            print_error("Cannot specify both --trace and --no-trace")
            raise typer.Exit(1)

        if not compiler and not runtime:
            targets = ["full"]
        else:
            targets = []
            if compiler:
                targets.append("compiler")
            if runtime:
                targets.append("runtime")

        features = ["driver", "metrics"]
        if no_trace:
            features.append("no-trace")
            print_header("Building APXM (no-trace)")
            print_info("Tracing disabled - zero overhead build")
        else:
            print_header("Building APXM")
            if trace:
                print_info("Tracing enabled - use --trace flag at runtime")

        if clean:
            print_step("Cleaning build artifacts...")
            build_dir = config.target_dir / "release" / "build"
            if build_dir.exists():
                for pattern in ["apxm-*"]:
                    for p in build_dir.glob(pattern):
                        shutil.rmtree(p, ignore_errors=True)
            print_success("Build artifacts cleaned")

        for target in targets:
            if target == "full":
                print_step("Building full project...")
                cmd = ["cargo", "build", "-p", "apxm-cli", "--features", ",".join(features)]
            elif target == "compiler":
                print_step("Building compiler...")
                cmd = ["cargo", "build", "-p", "apxm-compiler"]
            elif target == "runtime":
                print_step("Building runtime...")
                runtime_features = []
                if no_trace:
                    runtime_features.append("no-trace")
                if runtime_features:
                    cmd = [
                        "cargo",
                        "build",
                        "-p",
                        "apxm-runtime",
                        "--features",
                        ",".join(runtime_features),
                    ]
                else:
                    cmd = ["cargo", "build", "-p", "apxm-runtime"]

            if release:
                cmd.append("--release")

            result = subprocess.run(cmd, cwd=config.apxm_dir, env=env)

            if result.returncode != 0:
                print_error(f"Build failed: {target}")
                raise typer.Exit(1)

            print_success(f"Built: {target}")

        print_success("Build complete!")
        if "full" in targets or "compiler" in targets:
            print_info(f"Binary: {config.compiler_bin}")
        if no_trace:
            print_warning("Note: --trace flag at runtime will have no effect")
