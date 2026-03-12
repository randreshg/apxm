"""APXM configuration and path detection.

All environment setup (conda, MLIR, LD_LIBRARY_PATH) is handled by sniff's
auto_activate mechanism via .sniff.toml. This module only provides project
path detection for locating the Rust binary and workspace root.
"""

from dataclasses import dataclass
from pathlib import Path


def _find_project_root(start: Path, marker: str = "Cargo.toml") -> Path | None:
    """Walk up from *start* looking for a directory containing *marker*."""
    for parent in (start, *start.parents):
        if (parent / marker).exists():
            return parent
    return None


@dataclass
class ApxmConfig:
    """APXM project paths."""

    apxm_dir: Path
    target_dir: Path

    @classmethod
    def detect(cls) -> "ApxmConfig":
        """Auto-detect APXM project root from this file's location."""
        start = Path(__file__).parent.resolve()

        apxm_dir = _find_project_root(start)

        # Fallback: if we're in tools/, the parent is the project root
        if apxm_dir is None and start.name == "tools":
            apxm_dir = start.parent

        # Last resort: use the start directory itself
        if apxm_dir is None:
            apxm_dir = start

        return cls(apxm_dir=apxm_dir, target_dir=apxm_dir / "target")

    @property
    def compiler_bin(self) -> Path:
        """Path to the compiled apxm binary."""
        return self.target_dir / "release" / "apxm"
