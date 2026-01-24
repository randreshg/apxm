#!/usr/bin/env python3
"""
ArtsMate CLI - Development wrapper for ArtsMate.

Handles environment setup, building, and running the ArtsMate CLI.

Usage:
    artsmate              # Start interactive chat (default)
    artsmate build        # Build artsmate binary
    artsmate build --deps # Also rebuild apxm dependencies
    artsmate doctor       # Check environment status
"""

import subprocess
from pathlib import Path
from typing import Optional

import typer

from apxm_env import ApxmConfig, get_conda_prefix, setup_mlir_environment
from apxm_styles import console, print_error, print_header, print_info, print_step, print_success, print_warning

app = typer.Typer(
    name="artsmate",
    help="ArtsMate CLI - Development wrapper",
    invoke_without_command=True,
)


def get_config() -> ApxmConfig:
    """Get APXM configuration."""
    return ApxmConfig.detect()


def get_artsmate_dir() -> Path:
    """Get path to ArtsMate repository."""
    config = get_config()
    # ArtsMate is sibling to apxm
    return config.apxm_dir.parent / "ArtsMate"


def get_artsmate_rs_dir() -> Path:
    """Get path to artsmate-rs package."""
    return get_artsmate_dir() / "packages" / "artsmate-rs"


def get_binary_path(release: bool = False) -> Path:
    """Get path to artsmate binary."""
    mode = "release" if release else "debug"
    return get_artsmate_rs_dir() / "target" / mode / "artsmate"


def ensure_conda_env() -> Path:
    """Ensure conda environment is available."""
    conda_prefix = get_conda_prefix()
    if not conda_prefix:
        print_error("Conda environment 'apxm' not found!")
        print_info("Create it with: mamba env create -f environment.yaml")
        raise typer.Exit(1)
    return conda_prefix


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """ArtsMate CLI - Start chat or run subcommands."""
    if ctx.invoked_subcommand is None:
        # Default: run chat
        run_chat()


def run_chat(args: list[str] = None):
    """Run the artsmate chat interface."""
    config = get_config()
    conda_prefix = ensure_conda_env()
    env = setup_mlir_environment(conda_prefix, config.target_dir)

    binary = get_binary_path()
    if not binary.exists():
        print_error("Binary not built!")
        print_info("Run: artsmate build")
        raise typer.Exit(1)

    cmd = [str(binary)]
    if args:
        cmd.extend(args)

    result = subprocess.run(cmd, env=env)
    raise typer.Exit(result.returncode)


@app.command()
def build(
    deps: bool = typer.Option(False, "--deps", "-d", help="Also rebuild apxm dependencies"),
    release: bool = typer.Option(False, "--release", "-r", help="Build in release mode"),
):
    """Build the artsmate binary."""
    config = get_config()
    conda_prefix = ensure_conda_env()
    env = setup_mlir_environment(conda_prefix, config.target_dir)

    artsmate_rs = get_artsmate_rs_dir()
    if not artsmate_rs.exists():
        print_error(f"ArtsMate not found at {artsmate_rs}")
        raise typer.Exit(1)

    print_header("Building ArtsMate")

    if deps:
        print_step("Building apxm dependencies...")
        cmd = ["cargo", "build", "-p", "apxm-compiler", "-p", "apxm-runtime"]
        if release:
            cmd.append("--release")
        result = subprocess.run(cmd, cwd=config.apxm_dir, env=env)
        if result.returncode != 0:
            print_error("Failed to build dependencies")
            raise typer.Exit(1)
        print_success("Dependencies built")

    print_step("Building artsmate...")
    cmd = ["cargo", "build"]
    if release:
        cmd.append("--release")

    result = subprocess.run(cmd, cwd=artsmate_rs, env=env)

    if result.returncode == 0:
        binary = get_binary_path(release)
        print_success(f"Build complete: {binary}")
    else:
        print_error("Build failed!")
        raise typer.Exit(1)


@app.command()
def doctor():
    """Check environment status."""
    config = get_config()
    artsmate_dir = get_artsmate_dir()
    artsmate_rs = get_artsmate_rs_dir()

    print_header("ArtsMate Environment Check")

    # Check directories
    if config.apxm_dir.exists():
        print_success(f"APXM directory: {config.apxm_dir}")
    else:
        print_error(f"APXM directory not found: {config.apxm_dir}")

    if artsmate_dir.exists():
        print_success(f"ArtsMate directory: {artsmate_dir}")
    else:
        print_error(f"ArtsMate directory not found: {artsmate_dir}")

    # Check conda
    conda_prefix = get_conda_prefix()
    if conda_prefix:
        print_success(f"Conda environment: {conda_prefix}")

        # Check MLIR
        mlir_dir = conda_prefix / "lib" / "cmake" / "mlir"
        if mlir_dir.exists():
            print_success("MLIR: found")
        else:
            print_error("MLIR not found in conda env")
    else:
        print_error("Conda environment 'apxm' not found")

    # Check binaries
    for mode in ["debug", "release"]:
        binary = artsmate_rs / "target" / mode / "artsmate"
        if binary.exists():
            print_success(f"Binary ({mode}): {binary}")

    if not (artsmate_rs / "target" / "debug" / "artsmate").exists():
        print_warning("Binary not built (run: artsmate build)")

    # Check workflows
    workflows_dir = artsmate_dir / ".artsmate" / "workflows"
    if workflows_dir.exists():
        count = len(list(workflows_dir.glob("*.ais")))
        print_success(f"Workflows: {count} found")
    else:
        print_warning(f"Workflows directory not found: {workflows_dir}")

    console.print()


@app.command()
def chat(
    args: Optional[list[str]] = typer.Argument(None, help="Arguments to pass to artsmate"),
):
    """Start the interactive chat interface."""
    run_chat(args or [])


if __name__ == "__main__":
    app()
