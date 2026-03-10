"""Platform configuration detection for APXM CLI."""

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional


@dataclass
class PlatformConfig:
    """Detected platform configuration."""

    os_name: str  # "linux", "darwin", "windows"
    arch: str  # "x86_64", "aarch64"
    distro: Optional[str] = None  # "ubuntu", "fedora", etc. (Linux only)
    distro_version: Optional[str] = None
    pkg_manager: Optional[str] = None  # "apt", "dnf", "brew", "pacman"
    is_wsl: bool = False
    conda_cmd: Optional[str] = None  # "mamba" or "conda"
    has_rustup: bool = False

    @property
    def is_linux(self) -> bool:
        return self.os_name == "linux"

    @property
    def is_macos(self) -> bool:
        return self.os_name == "darwin"


def _detect_linux_distro() -> tuple[Optional[str], Optional[str]]:
    """Detect Linux distribution from /etc/os-release."""
    try:
        with open("/etc/os-release") as f:
            data = {}
            for line in f:
                if "=" in line:
                    key, _, value = line.strip().partition("=")
                    data[key] = value.strip('"')
            return data.get("ID"), data.get("VERSION_ID")
    except FileNotFoundError:
        return None, None


def _detect_pkg_manager() -> Optional[str]:
    """Detect the system package manager."""
    for cmd in ["apt", "dnf", "yum", "pacman", "brew", "zypper"]:
        if shutil.which(cmd):
            return cmd
    return None


def _detect_conda() -> Optional[str]:
    """Detect conda/mamba command."""
    if shutil.which("mamba"):
        return "mamba"
    if shutil.which("conda"):
        return "conda"
    return None


def _detect_wsl() -> bool:
    """Detect if running in WSL."""
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except FileNotFoundError:
        return False


@lru_cache(maxsize=1)
def get_platform_config() -> PlatformConfig:
    """Get the platform configuration (cached singleton)."""
    os_name = platform.system().lower()
    arch = platform.machine()

    distro = None
    distro_version = None
    is_wsl = False

    if os_name == "linux":
        distro, distro_version = _detect_linux_distro()
        is_wsl = _detect_wsl()

    return PlatformConfig(
        os_name=os_name,
        arch=arch,
        distro=distro,
        distro_version=distro_version,
        pkg_manager=_detect_pkg_manager(),
        is_wsl=is_wsl,
        conda_cmd=_detect_conda(),
        has_rustup=shutil.which("rustup") is not None,
    )
