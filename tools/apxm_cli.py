#!/usr/bin/env python3
"""
APXM CLI - Python driver for APXM compiler and runtime.

Automatically handles conda environment detection, MLIR environment setup,
and provides convenient commands for building, running, and testing.

Usage:
    apxm doctor                     # Check environment status (built-in via sniff)
    apxm version                    # Show version information (built-in via sniff)
    apxm build                      # Build compiler and runtime
    apxm execute workflow.json      # Compile and execute an ApxmGraph file
    apxm compile workflow.json -o out.apxmobj  # Compile to artifact
    apxm run out.apxmobj            # Run pre-compiled artifact
    apxm test                       # Run test suite
    apxm install                    # Install/update environment
    apxm register add <name> ...    # Register API credentials
    apxm --help                     # Show all available commands
"""

from typing import Optional

from sniff import Typer, Option, Argument, Exit

from scripts.build import register_commands as register_build
from scripts.compile import register_commands as register_compile
from scripts.execute import register_commands as register_execute
from scripts.install import register_commands as register_install
from scripts.register import create_app as create_register_app
from scripts.run import register_commands as register_run
from scripts.test import register_commands as register_test

VERSION = "0.2.0"

# Main CLI app with auto-activation
# This automatically activates the conda environment from .sniff.toml
# before running commands (except install, which handles it specially)
app = Typer(
    name="apxm",
    auto_activate=True,  # Auto-detect and activate environment
    fail_fast=False,     # Don't fail immediately - let commands handle it
    add_doctor_command=True,
    add_version_command=True,
    project_version=VERSION,
    help="APXM CLI - Compiler and runtime driver",
    no_args_is_help=True,
)

# Register top-level commands
register_build(app)
register_compile(app)
register_execute(app)
register_run(app)
register_test(app)
register_install(app)

# Register sub-apps
app.add_typer(create_register_app(), name="register")


if __name__ == "__main__":
    app()
