"""Dependency checking for APXM CLI -- powered by sniff."""

from __future__ import annotations

from dataclasses import dataclass

from sniff import DependencyChecker, DependencySpec, DependencyResult
from sniff.version import VersionSpec, version_satisfies

from .messages import (
    FIX_SUGGESTIONS,
    MSG_DEP_INSTALL_GENERIC,
    MSG_DEP_UPGRADE_GENERIC,
)


@dataclass
class ApxmDependency:
    """An APXM dependency: a sniff DependencySpec plus a rich version constraint."""

    spec: DependencySpec
    version_constraint: VersionSpec | None = None

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def command(self) -> str:
        return self.spec.command


def _dep(
    name: str,
    command: str,
    version_pattern: str | None = None,
    version_constraint: str | None = None,
    required: bool = True,
    fallback_commands: list[str] | None = None,
) -> ApxmDependency:
    """Build an ApxmDependency with a parsed VersionSpec constraint."""
    spec = DependencySpec(
        name=name,
        command=command,
        version_pattern=version_pattern,
        min_version=None,
        required=required,
        fallback_commands=fallback_commands,
    )
    parsed = VersionSpec.parse(version_constraint) if version_constraint else None
    return ApxmDependency(spec=spec, version_constraint=parsed)


# Standard APXM dependencies with version constraints
APXM_DEPENDENCIES: list[ApxmDependency] = [
    _dep(
        name="Rust (nightly)",
        command="rustc",
        version_pattern=r"rustc (\d+\.\d+\.\d+)",
        version_constraint=">=1.80",
    ),
    _dep(
        name="Cargo",
        command="cargo",
        version_pattern=r"cargo (\d+\.\d+\.\d+)",
        version_constraint=">=1.80",
    ),
    _dep(
        name="Mamba/Conda",
        command="mamba",
        version_pattern=r"(\d+\.\d+\.\d+)",
        fallback_commands=["conda"],
    ),
    _dep(
        name="CMake",
        command="cmake",
        version_pattern=r"cmake version (\d+\.\d+\.\d+)",
        version_constraint=">=3.20",
    ),
    _dep(
        name="Ninja",
        command="ninja",
        version_pattern=r"(\d+\.\d+\.\d+)",
        required=False,
    ),
    _dep(
        name="Git",
        command="git",
        version_pattern=r"git version (\d+\.\d+\.\d+)",
    ),
    _dep(
        name="LLVM (21+)",
        command="llvm-config",
        version_pattern=r"(\d+\.\d+\.\d+)",
        version_constraint=">=21.0",
        required=False,
    ),
]


def _check_one(dep: ApxmDependency, checker: DependencyChecker) -> DependencyResult:
    """Check a single dependency using DependencyChecker + VersionSpec."""
    result = checker.check(dep.spec)

    if not result.found or not result.version or dep.version_constraint is None:
        return result

    meets = version_satisfies(result.version, dep.version_constraint.raw)

    if meets == result.meets_minimum:
        return result

    return DependencyResult(
        name=result.name,
        command=result.command,
        found=result.found,
        version=result.version,
        meets_minimum=meets,
        required=result.required,
        error=result.error,
    )


def check_all(
    deps: list[ApxmDependency] | None = None,
) -> list[DependencyResult]:
    """Check all APXM dependencies and return results.

    Args:
        deps: Optional list of ApxmDependency to check. Defaults to APXM_DEPENDENCIES.

    Returns:
        List of DependencyResult from sniff.
    """
    if deps is None:
        deps = APXM_DEPENDENCIES
    checker = DependencyChecker()
    return [_check_one(dep, checker) for dep in deps]


def get_fix_suggestion(result: DependencyResult) -> str | None:
    """Get a fix suggestion for a failed dependency check.

    Args:
        result: The DependencyResult to suggest a fix for.

    Returns:
        A human-readable fix suggestion, or None.
    """
    if result.ok:
        return None

    suggestion = FIX_SUGGESTIONS.get(result.command)
    if suggestion:
        return suggestion

    if not result.found:
        return MSG_DEP_INSTALL_GENERIC.format(name=result.name)
    if not result.meets_minimum:
        return MSG_DEP_UPGRADE_GENERIC.format(name=result.name)
    return None
