"""Test command for APXM CLI."""

import subprocess
from typing import Optional

import typer

from apxm_env import setup_mlir_environment
from apxm_styles import print_error, print_header, print_info, print_success, print_warning

from tools.scripts import ensure_conda_env, get_config


def register_commands(app: typer.Typer) -> None:
    """Register test command on the app."""

    @app.command()
    def test(
        all: bool = typer.Option(
            False, "--all", help="Run all tests including compiler (requires MLIR/LLVM 21)"
        ),
        runtime: bool = typer.Option(False, "--runtime", help="Run only runtime tests"),
        compiler: bool = typer.Option(
            False, "--compiler", help="Run only compiler tests (requires MLIR/LLVM 21)"
        ),
        credentials: bool = typer.Option(
            False, "--credentials", help="Run only credential store tests"
        ),
        backends: bool = typer.Option(
            False, "--backends", help="Run only LLM backend tests (uses mocks, no API keys needed)"
        ),
        package: Optional[str] = typer.Option(
            None, "--package", "-p", help="Run tests for a specific package"
        ),
    ):
        """Run APXM test suite.

        By default, runs workspace tests excluding the compiler (which requires
        MLIR/LLVM 21). No LLM API keys are needed -- all backend tests use mocks.

        The only distinction is compiler tests (need MLIR installed) vs everything
        else (always works offline). The test suite uses MockLLMBackend and dummy
        keys throughout -- no real API calls are made.

        Examples:
            apxm test                    # All tests except compiler (no API keys needed)
            apxm test --all              # All tests (requires MLIR/LLVM 21)
            apxm test --runtime          # Runtime tests only
            apxm test --compiler         # Compiler tests only (requires MLIR)
            apxm test --credentials      # Credential store tests only
            apxm test --backends         # Backend mock tests only
            apxm test --package <name>   # Specific package tests
        """
        config = get_config()
        conda_prefix = ensure_conda_env()
        env = setup_mlir_environment(conda_prefix, config.target_dir)

        cmd = ["cargo", "test"]

        if package:
            print_header(f"Testing {package}")
            cmd.extend(["-p", package])
        elif credentials:
            print_header("Testing Credentials")
            cmd.extend(["-p", "apxm-credentials"])
        elif backends:
            print_header("Testing Backends (mocks)")
            print_info("All backend tests use MockLLMBackend -- no API keys needed")
            cmd.extend(["-p", "apxm-backends"])
        elif compiler:
            print_header("Testing Compiler (requires MLIR)")
            cmd.extend(["-p", "apxm-compiler"])
        elif runtime:
            print_header("Testing Runtime")
            cmd.extend(["-p", "apxm-runtime"])
        elif all:
            print_header("Testing All (workspace)")
            print_warning("Includes compiler tests -- requires MLIR/LLVM 21")
            cmd.append("--workspace")
        else:
            print_header("Testing (excluding compiler)")
            print_info("No API keys needed -- all tests use mocks")
            cmd.extend(["--workspace", "--exclude", "apxm-compiler"])

        result = subprocess.run(cmd, cwd=config.apxm_dir, env=env)

        if result.returncode == 0:
            print_success("All tests passed!")
        else:
            print_error("Some tests failed!")

        raise typer.Exit(result.returncode)
