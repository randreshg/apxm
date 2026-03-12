"""CI environment detection and Cargo build tuning for APXM.

Thin Cargo-specific adapter over sniff.ci_build.CIBuildAdvisor.
All CI detection logic and parallelism heuristics live in sniff;
this module only maps build-system-agnostic hints to Cargo flags.
"""

from __future__ import annotations

from sniff.ci import CIBuildAdvisor, CIBuildHints, CIDetector, CIInfo


def detect_ci() -> CIInfo:
    """Detect the CI environment using sniff."""
    return CIDetector().detect()


def ci_build_hints(ci: CIInfo | None = None) -> CIBuildHints:
    """Get build hints for the current environment via CIBuildAdvisor."""
    if ci is None:
        ci = detect_ci()
    return CIBuildAdvisor(ci).advise()


def apply_ci_cargo_flags(cmd: list[str], hints: CIBuildHints) -> list[str]:
    """Append CI-tuned flags to a cargo command."""
    result = list(cmd)
    if not hints.ci_output:
        return result
    if "--locked" not in result:
        result.append("--locked")
    if hints.max_jobs is not None and "--jobs" not in " ".join(result):
        result.append(f"--jobs={hints.max_jobs}")
    return result


def apply_ci_test_flags(cmd: list[str], hints: CIBuildHints) -> list[str]:
    """Append CI-tuned test flags (after '--') to a cargo test command."""
    result = list(cmd)
    if hints.max_test_workers is None:
        return result
    if "--" not in result:
        result.append("--")
    result.append(f"--test-threads={hints.max_test_workers}")
    return result


def apply_ci_env(env: dict[str, str], hints: CIBuildHints) -> dict[str, str]:
    """Merge CI environment overrides into the build environment."""
    result = dict(env)
    if not hints.incremental:
        result["CARGO_INCREMENTAL"] = "0"
    if hints.use_color:
        result["CARGO_TERM_COLOR"] = "always"
    result.update(hints.env_hints)
    return result
