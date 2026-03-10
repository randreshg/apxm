"""Compiler sub-commands for APXM CLI."""

import shutil
import subprocess

import typer

from apxm_env import setup_mlir_environment
from apxm_styles import (
    print_error,
    print_header,
    print_info,
    print_step,
    print_success,
    print_warning,
)

from tools.scripts import ensure_conda_env, get_config


def create_app() -> typer.Typer:
    """Create the compiler sub-app."""
    compiler_app = typer.Typer(help="Compiler operations")

    @compiler_app.command("build")
    def compiler_build(
        release: bool = typer.Option(True, "--release/--debug", help="Build in release mode"),
        clean: bool = typer.Option(
            False, "--clean", "-c", help="Clean build artifacts first"
        ),
        trace: bool = typer.Option(
            False, "--trace", help="Build with tracing enabled (default)"
        ),
        no_trace: bool = typer.Option(
            False, "--no-trace", help="Build with tracing disabled (zero overhead)"
        ),
    ):
        """Build the APXM compiler.

        Tracing control:
          --trace     Build with runtime tracing enabled (default)
          --no-trace  Build with tracing compiled out (zero overhead)
        """
        config = get_config()
        conda_prefix = ensure_conda_env()
        env = setup_mlir_environment(conda_prefix, config.target_dir)

        if trace and no_trace:
            print_error("Cannot specify both --trace and --no-trace")
            raise typer.Exit(1)

        features = ["driver", "metrics"]
        if no_trace:
            features.append("no-trace")
            print_header("Building APXM Compiler (no-trace)")
            print_info("Tracing disabled - zero overhead build")
        else:
            print_header("Building APXM Compiler")

        if clean:
            print_step("Cleaning build artifacts...")
            build_dir = config.target_dir / "release" / "build"
            if build_dir.exists():
                for pattern in ["apxm-compiler-*", "apxm-driver-*"]:
                    for p in build_dir.glob(pattern):
                        shutil.rmtree(p, ignore_errors=True)
            print_success("Build artifacts cleaned")

        print_step("Building compiler (this may take a while)...")

        cmd = ["cargo", "build", "-p", "apxm-cli", "--features", ",".join(features)]
        if release:
            cmd.append("--release")

        result = subprocess.run(cmd, cwd=config.apxm_dir, env=env)

        if result.returncode == 0:
            print_success("Compiler built successfully!")
            print_info(f"Binary: {config.compiler_bin}")
            if no_trace:
                print_warning("Note: --trace flag at runtime will have no effect")
        else:
            print_error("Build failed!")
            raise typer.Exit(1)

    return compiler_app
