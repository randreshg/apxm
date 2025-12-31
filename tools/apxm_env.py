"""
APXM Environment Utilities

Shared environment detection and setup utilities used by both apxm_cli.py
and benchmark runner scripts.

This module provides:
- get_conda_prefix() - Find the apxm conda environment
- setup_mlir_environment() - Set up MLIR/LLVM environment variables
- ApxmConfig - Centralized configuration dataclass
"""

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def get_conda_prefix() -> Optional[Path]:
    """Find the apxm conda environment.

    Checks in order:
    1. CONDA_PREFIX environment variable (if it's the apxm env or has MLIR)
    2. Query conda for environment list
    3. Query mamba for environment list

    Returns:
        Path to the conda environment, or None if not found.
    """
    # 1. Check CONDA_PREFIX if already activated
    if "CONDA_PREFIX" in os.environ:
        prefix = Path(os.environ["CONDA_PREFIX"])
        if prefix.name == "apxm" or (prefix / "lib" / "cmake" / "mlir").exists():
            return prefix

    # 2. Query conda for environment list
    try:
        result = subprocess.run(
            ["conda", "info", "--envs", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        for env_path in data.get("envs", []):
            if Path(env_path).name == "apxm":
                return Path(env_path)
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        pass

    # 3. Try mamba
    try:
        result = subprocess.run(
            ["mamba", "info", "--envs", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        for env_path in data.get("envs", []):
            if Path(env_path).name == "apxm":
                return Path(env_path)
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        pass

    return None


def setup_mlir_environment(conda_prefix: Path, target_dir: Optional[Path] = None) -> dict[str, str]:
    """Configure environment for MLIR compilation and execution.

    Sets up:
    - CONDA_PREFIX to the correct environment
    - MLIR_DIR, LLVM_DIR for compilation
    - DYLD_LIBRARY_PATH (macOS) or LD_LIBRARY_PATH (Linux) for runtime

    Args:
        conda_prefix: Path to the conda environment
        target_dir: Optional target directory (for release libraries)

    Returns:
        Dictionary of environment variables to use.
    """
    env = os.environ.copy()

    # Set CONDA_PREFIX to the correct environment (critical for Rust runtime)
    env["CONDA_PREFIX"] = str(conda_prefix)
    env["MLIR_DIR"] = str(conda_prefix / "lib" / "cmake" / "mlir")
    env["LLVM_DIR"] = str(conda_prefix / "lib" / "cmake" / "llvm")

    if target_dir:
        # Add both release/ and release/lib/ for dylib lookup
        target_release = target_dir / "release"
        paths = [str(target_release / "lib"), str(target_release)]
    else:
        paths = []

    lib_path = str(conda_prefix / "lib")
    paths.append(lib_path)

    if platform.system() == "Darwin":
        existing = env.get("DYLD_LIBRARY_PATH", "")
        if existing:
            paths.append(existing)
        env["DYLD_LIBRARY_PATH"] = ":".join(paths)
    else:
        existing = env.get("LD_LIBRARY_PATH", "")
        if existing:
            paths.append(existing)
        env["LD_LIBRARY_PATH"] = ":".join(paths)

    return env


@dataclass
class ApxmConfig:
    """APXM configuration and paths."""

    apxm_dir: Path
    conda_prefix: Optional[Path]
    target_dir: Path

    @classmethod
    def detect(cls) -> "ApxmConfig":
        """Auto-detect APXM configuration.

        Tries to find the APXM root directory by looking for common markers
        (Cargo.toml, tools/, etc.) starting from this file's location.
        """
        # Start from this file's directory and find apxm root
        current = Path(__file__).parent.resolve()

        # If we're in tools/, go up one level
        if current.name == "tools":
            apxm_dir = current.parent
        else:
            # Try to find apxm root by looking for markers
            apxm_dir = current
            while apxm_dir.parent != apxm_dir:
                if (apxm_dir / "Cargo.toml").exists() and (apxm_dir / "tools").exists():
                    break
                apxm_dir = apxm_dir.parent

        return cls(
            apxm_dir=apxm_dir,
            conda_prefix=get_conda_prefix(),
            target_dir=apxm_dir / "target",
        )

    @property
    def compiler_bin(self) -> Path:
        """Path to the compiled apxm binary."""
        return self.target_dir / "release" / "apxm"

    @property
    def workloads_dir(self) -> Path:
        """Path to benchmark workloads."""
        return self.apxm_dir / "papers" / "cf26" / "benchmarks" / "workloads"

    def get_mlir_env(self) -> dict[str, str]:
        """Get environment variables for MLIR compilation/execution."""
        if self.conda_prefix is None:
            return os.environ.copy()
        return setup_mlir_environment(self.conda_prefix, self.target_dir)
