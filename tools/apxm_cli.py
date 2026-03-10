#!/usr/bin/env python3
"""
APXM CLI - Python driver for APXM compiler and runtime.

Automatically handles conda environment detection, MLIR environment setup,
and provides convenient commands for building, running, and testing.

Usage:
    apxm doctor                     # Check environment status
    apxm build                      # Build compiler and runtime
    apxm execute workflow.json      # Compile and execute an ApxmGraph file
    apxm compile workflow.json -o out.apxmobj  # Compile to artifact
    apxm run out.apxmobj            # Run pre-compiled artifact
    apxm test                       # Run test suite
    apxm install                    # Install/update environment
    apxm register add <name> ...    # Register API credentials
"""

import typer

from scripts.build import register_commands as register_build
from scripts.compile import register_commands as register_compile
from scripts.compiler import create_app as create_compiler_app
from scripts.doctor import register_commands as register_doctor
from scripts.execute import register_commands as register_execute
from scripts.install import register_commands as register_install
from scripts.register import create_app as create_register_app
from scripts.run import register_commands as register_run
from scripts.test import register_commands as register_test

# Main CLI app
app = typer.Typer(
    name="apxm-cli",
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
register_doctor(app)

# Register sub-apps
app.add_typer(create_compiler_app(), name="compiler")
app.add_typer(create_register_app(), name="register")


if __name__ == "__main__":
    app()
