"""Test command for APXM CLI."""

import os
import subprocess
from typing import Optional

from sniff import Typer, Option, Exit
from sniff import print_error, print_header, print_info, print_success, print_warning

from . import get_config
from .ci_env import (
    apply_ci_cargo_flags,
    apply_ci_env,
    apply_ci_test_flags,
    ci_build_hints,
)


def register_commands(app: Typer) -> None:
    """Register test command on the app."""

    @app.command()
    def test(
        all: bool = Option(
            False, "--all", help="Run all tests including compiler (requires MLIR/LLVM 21)"
        ),
        runtime: bool = Option(False, "--runtime", help="Run only runtime tests"),
        compiler: bool = Option(
            False, "--compiler", help="Run only compiler tests (requires MLIR/LLVM 21)"
        ),
        credentials: bool = Option(
            False, "--credentials", help="Run only credential store tests"
        ),
        backends: bool = Option(
            False, "--backends", help="Run only LLM backend tests (uses mocks, no API keys needed)"
        ),
        package: Optional[str] = Option(
            None, "--package", "-p", help="Run tests for a specific package"
        ),
    ):
        """Run APXM test suite.

        By default, runs workspace tests excluding the compiler (which requires
        MLIR/LLVM 21). No LLM API keys are needed -- all backend tests use mocks.

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

        # CI environment overrides
        ci = app.ci_info
        hints = ci_build_hints(ci)
        env = None
        if ci.is_ci:
            env = apply_ci_env(dict(os.environ), hints)
            provider = ci.provider.display_name if ci.provider else "Unknown CI"
            print_info(f"CI detected: {provider}")

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

        cmd = apply_ci_cargo_flags(cmd, hints)
        cmd = apply_ci_test_flags(cmd, hints)

        result = subprocess.run(cmd, cwd=config.apxm_dir, env=env)

        if result.returncode == 0:
            print_success("All tests passed!")
        else:
            print_error("Some tests failed!")

        raise Exit(result.returncode)
