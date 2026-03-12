"""APXM CLI Scripts - Modular command implementations.

Environment setup (conda, MLIR, LD_LIBRARY_PATH) is handled by sniff's
auto_activate before any command runs. Scripts just need project paths.
"""

import subprocess
from pathlib import Path

from sniff import Exit
from sniff import print_error, print_info

from apxm_env import ApxmConfig
from . import messages as msg

CARGO_FEATURES = "driver,metrics"


def get_config() -> ApxmConfig:
    """Get the APXM configuration."""
    return ApxmConfig.detect()


def ensure_binary(config: ApxmConfig) -> None:
    """Check that the APXM binary is built. Raises Exit(1) if missing."""
    if not config.compiler_bin.exists():
        print_error("APXM binary not built!")
        print_info("Run: apxm build")
        raise Exit(1)


def build_apxm_cmd(config: ApxmConfig, subcommand: str, extra_args: list[str],
                    cargo: bool = False) -> list[str]:
    """Build the command list for running an apxm subcommand."""
    if cargo:
        return [
            "cargo", "run", "-p", "apxm-cli",
            "--features", CARGO_FEATURES, "--release",
            "--", subcommand, *extra_args,
        ]
    ensure_binary(config)
    return [str(config.compiler_bin), subcommand, *extra_args]


def resolve_path(path: "Path | None", cwd: Path) -> "Path | None":
    """Resolve an optional path relative to cwd."""
    if path is not None and not path.is_absolute():
        return (cwd / path).resolve()
    return path


def run_apxm(config: ApxmConfig, cmd: list[str], env: dict | None = None) -> int:
    """Run a command in the APXM directory and return the exit code."""
    result = subprocess.run(cmd, cwd=config.apxm_dir, env=env)
    return result.returncode
