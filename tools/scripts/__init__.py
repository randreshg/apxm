"""APXM CLI Scripts - Modular command implementations."""

from pathlib import Path

import typer

from apxm_env import ApxmConfig, get_conda_prefix
from apxm_styles import print_error, print_info


def get_config() -> ApxmConfig:
    """Get the APXM configuration."""
    return ApxmConfig.detect()


def ensure_conda_env() -> Path:
    """Verify conda environment is activated."""
    conda_prefix = get_conda_prefix()
    if not conda_prefix:
        print_error("Conda environment 'apxm' not found!")
        print_info("Create it with: cargo run -p apxm-cli -- install")
        raise typer.Exit(1)
    return conda_prefix
