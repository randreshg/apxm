"""Dependency checking for APXM CLI."""

import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class Dependency:
    """A system dependency to check."""

    name: str
    command: str
    version_arg: str = "--version"
    version_pattern: Optional[str] = None  # regex to extract version
    min_version: Optional[str] = None
    required: bool = True


@dataclass
class CheckResult:
    """Result of checking a dependency."""

    dep: Dependency
    found: bool
    version: Optional[str] = None
    meets_minimum: bool = True
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.found and self.meets_minimum


# Standard APXM dependencies
DEPENDENCIES = [
    Dependency(
        name="Rust (nightly)",
        command="rustc",
        version_pattern=r"rustc (\d+\.\d+\.\d+)",
    ),
    Dependency(
        name="Cargo",
        command="cargo",
        version_pattern=r"cargo (\d+\.\d+\.\d+)",
    ),
    Dependency(
        name="Mamba/Conda",
        command="mamba",
        version_pattern=r"(\d+\.\d+\.\d+)",
    ),
    Dependency(
        name="CMake",
        command="cmake",
        version_pattern=r"cmake version (\d+\.\d+\.\d+)",
        min_version="3.20",
    ),
    Dependency(
        name="Ninja",
        command="ninja",
        version_pattern=r"(\d+\.\d+\.\d+)",
        required=False,
    ),
    Dependency(
        name="Git",
        command="git",
        version_pattern=r"git version (\d+\.\d+\.\d+)",
    ),
]


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse a version string into a tuple of ints for comparison."""
    parts = []
    for part in version_str.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)


def _version_ge(version: str, minimum: str) -> bool:
    """Check if version >= minimum."""
    return _parse_version(version) >= _parse_version(minimum)


def check_dependency(dep: Dependency) -> CheckResult:
    """Check if a dependency is available and meets version requirements."""
    cmd_path = shutil.which(dep.command)

    # Fallback for mamba -> conda
    if not cmd_path and dep.command == "mamba":
        cmd_path = shutil.which("conda")

    if not cmd_path:
        return CheckResult(
            dep=dep,
            found=False,
            error=f"{dep.name} not found in PATH",
        )

    version = None
    if dep.version_pattern:
        try:
            result = subprocess.run(
                [cmd_path, dep.version_arg],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout + result.stderr
            match = re.search(dep.version_pattern, output)
            if match:
                version = match.group(1)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    meets_min = True
    if version and dep.min_version:
        meets_min = _version_ge(version, dep.min_version)

    return CheckResult(
        dep=dep,
        found=True,
        version=version,
        meets_minimum=meets_min,
    )


def check_all(deps: Optional[list[Dependency]] = None) -> list[CheckResult]:
    """Check all dependencies and return results."""
    if deps is None:
        deps = DEPENDENCIES
    return [check_dependency(dep) for dep in deps]
