"""Enhanced install command for APXM CLI."""

import shutil
import subprocess
from pathlib import Path

from sniff import Typer, Option, Exit, BinaryInstaller

from apxm_env import check_conda_env, get_conda_prefix, setup_mlir_environment
from sniff import (
    print_blank,
    print_error,
    print_header,
    print_info,
    print_next_steps,
    print_numbered_list,
    print_step,
    print_success,
    print_warning,
    spinner,
)

from . import get_config, messages as msg
from .deps import check_all, get_fix_suggestion


def register_commands(app: Typer) -> None:
    """Register install command on the app."""

    @app.command()
    def install(
        check: bool = Option(False, "--check", help="Dry-run: report status without changes"),
        skip_deps: bool = Option(False, "--skip-deps", help="Skip dependency checks"),
        skip_build: bool = Option(False, "--skip-build", help="Skip build step"),
        auto: bool = Option(
            False, "--auto", "-y", help="Automatic mode (no prompts)"
        ),
    ):
        """Install or update APXM environment.

        Checks all dependencies, reports what's missing, and proceeds with
        whatever stages are possible. Never crashes on missing deps.

        Stages:
          1. Platform detection
          2. Dependency checks
          3. Conda environment
          4. Rust toolchain
          5. Build
          6. Summary
        """
        config = get_config()
        platform = app.platform
        missing: list[str] = []

        print_header(msg.HEADER_INSTALL)

        # ── Stage 1 ───────────────────────────────────────────────
        print_step(msg.STAGE_PLATFORM_DETECTION)
        os_label = f"{platform.os} ({platform.arch})"
        print_info(msg.MSG_OS.format(os_label=os_label))
        if platform.distro:
            print_info(msg.MSG_DISTRO.format(
                distro=platform.distro, version=platform.distro_version or "",
            ))
        if platform.is_wsl:
            print_info(msg.MSG_WSL_DETECTED)
        if platform.pkg_manager:
            print_info(msg.MSG_PKG_MANAGER.format(pkg_manager=platform.pkg_manager))
        print_blank()

        # ── Stage 2 ───────────────────────────────────────────────
        # Mamba/Conda and Rust are checked in detail by Stages 3 & 4,
        # so Stage 2 only reports their status without adding to missing.
        stage3_4_names = {"Mamba/Conda", "Rust (nightly)", "Cargo"}
        if not skip_deps:
            print_step(msg.STAGE_CHECKING_DEPS)
            results = check_all()
            for r in results:
                version_str = f" ({r.version})" if r.version else ""
                if r.found:
                    if r.meets_minimum:
                        print_success(msg.MSG_DEP_OK.format(name=r.name, version=version_str))
                    else:
                        print_warning(
                            msg.MSG_DEP_NEEDS_UPGRADE.format(name=r.name, version=version_str)
                        )
                        if r.name not in stage3_4_names:
                            missing.append(msg.MSG_DEP_UPGRADE_REQUIRED.format(name=r.name))
                else:
                    if r.required:
                        print_error(msg.MSG_DEP_NOT_FOUND.format(name=r.name))
                        if r.name not in stage3_4_names:
                            missing.append(r.name)
                    else:
                        print_warning(msg.MSG_DEP_NOT_FOUND_OPTIONAL.format(name=r.name))
            print_blank()

        # ── Stage 3 ───────────────────────────────────────────────
        print_step(msg.STAGE_CONDA_ENV)
        # Find mamba or conda
        conda_cmd = shutil.which("mamba") or shutil.which("conda")
        if not conda_cmd:
            print_error(msg.MSG_CONDA_NOT_FOUND)
            missing.append("Mamba/Conda")
            print_blank()
        else:
            print_success(msg.MSG_CONDA_FOUND.format(cmd=conda_cmd))

            env_yaml = config.apxm_dir / "environment.yaml"
            if not env_yaml.exists():
                print_error(msg.MSG_CONDA_ENV_YAML_NOT_FOUND.format(path=env_yaml))
                missing.append("environment.yaml")
            elif check:
                conda_check = check_conda_env()
                for w in conda_check.warnings:
                    print_warning(w)
                if conda_check.prefix:
                    print_success(msg.MSG_CONDA_ENV_OK.format(prefix=conda_check.prefix))
                else:
                    print_warning(msg.MSG_CONDA_ENV_NOT_FOUND_WILL_CREATE)
                    missing.append(msg.MSG_CONDA_ENV_NOT_FOUND_RUN_INSTALL)
            else:
                # Try create first (with spinner for better UX)
                with spinner("Creating conda environment..."):
                    result = subprocess.run(
                        [conda_cmd, "env", "create", "-f", str(env_yaml)],
                        capture_output=True,
                        text=True,
                    )

                if result.returncode != 0:
                    # Create failed, try update instead
                    with spinner("Updating conda environment..."):
                        result = subprocess.run(
                            [conda_cmd, "env", "update", "-f", str(env_yaml), "-n", "apxm"],
                            capture_output=True,
                            text=True,
                        )
                    if result.returncode != 0:
                        print_error(msg.MSG_CONDA_CREATE_FAILED.format(cmd=conda_cmd))
                        # Show error output for debugging
                        if result.stderr:
                            print_info(f"Error: {result.stderr[:200]}")
                        missing.append(msg.MSG_CONDA_ENV_CREATE_FAILED)
                    else:
                        print_success(msg.MSG_CONDA_ENV_UPDATED)
                else:
                    print_success(msg.MSG_CONDA_ENV_CREATED)
            print_blank()

        # ── Stage 4 ───────────────────────────────────────────────
        print_step(msg.STAGE_RUST_TOOLCHAIN)
        has_rustup = shutil.which("rustup") is not None
        if has_rustup:
            print_success(msg.MSG_RUSTUP_FOUND)
            if not check:
                try:
                    result = subprocess.run(
                        ["rustup", "show"],
                        capture_output=True, text=True, timeout=30,
                    )
                    if "nightly" in result.stdout:
                        print_success(msg.MSG_RUST_NIGHTLY_INSTALLED)
                    else:
                        with spinner("Installing Rust nightly toolchain..."):
                            subprocess.run(
                                ["rustup", "toolchain", "install", "nightly"],
                                capture_output=True, timeout=300,
                            )
                        print_success("Rust nightly installed")
                except (subprocess.TimeoutExpired, OSError) as e:
                    print_warning(msg.MSG_RUST_CHECK_FAILED.format(error=e))
            else:
                try:
                    result = subprocess.run(
                        ["rustc", "--version"],
                        capture_output=True, text=True, timeout=10,
                    )
                    version = result.stdout.strip()
                    if "nightly" in version:
                        print_success(msg.MSG_RUST_VERSION.format(version=version))
                    else:
                        print_warning(msg.MSG_RUST_NIGHTLY_RECOMMENDED.format(version=version))
                        missing.append(msg.MSG_RUST_NIGHTLY_MISSING)
                except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                    print_warning(msg.MSG_RUST_VERSION_UNKNOWN)
        else:
            print_error(msg.MSG_RUSTUP_NOT_FOUND)
            missing.append("rustup")
            if not check and auto:
                try:
                    with spinner("Installing rustup and Rust nightly..."):
                        subprocess.run(
                            [
                                "sh", "-c",
                                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly",
                            ],
                            capture_output=True, timeout=300,
                        )
                    print_success("Rustup installed")
                except (subprocess.TimeoutExpired, OSError) as e:
                    print_error(msg.MSG_RUSTUP_INSTALL_FAILED.format(error=e))
        print_blank()

        # ── Stage 5 ───────────────────────────────────────────────
        if not skip_build:
            print_step(msg.STAGE_BUILD)
            has_cargo = shutil.which("cargo") is not None
            conda_prefix = get_conda_prefix()

            if check:
                if config.compiler_bin.exists():
                    print_success(msg.MSG_COMPILER_BIN_OK_SHORT.format(path=config.compiler_bin))
                else:
                    print_warning(msg.MSG_COMPILER_NOT_BUILT_YET)
                    missing.append(msg.MSG_BUILD_CONDA_ACTIVATE)
            elif not has_cargo:
                print_warning(msg.MSG_CARGO_NOT_AVAILABLE)
                missing.append(msg.MSG_BUILD_INSTALL_RUST)
            elif not conda_prefix:
                print_warning(msg.MSG_CONDA_NOT_ACTIVATED)
                missing.append(msg.MSG_BUILD_CONDA_ACTIVATE)
            else:
                env = setup_mlir_environment(conda_prefix, config.target_dir)
                print_info("Building APXM (this may take a few minutes)...")
                result = subprocess.run(
                    [
                        "cargo", "build",
                        "-p", "apxm-cli",
                        "--features", "driver,metrics",
                        "--release",
                    ],
                    cwd=config.apxm_dir,
                    env=env,
                )
                if result.returncode == 0:
                    print_success(msg.MSG_BUILD_COMPLETE)

                    # Install binary to bin/
                    binary_path = config.apxm_dir / "target" / "release" / "apxm"
                    if binary_path.exists():
                        try:
                            installer = BinaryInstaller(config.apxm_dir)
                            result = installer.install_binary(binary_path, update_shell=True)
                            print_success(result.message)
                        except Exception as e:
                            print_warning(f"Binary install failed: {e}")
                else:
                    print_error(msg.MSG_BUILD_FAILED)
                    missing.append(msg.MSG_BUILD_FIX_RETRY)
        print_blank()

        # ── Stage 6: Summary ──────────────────────────────────────
        if missing:
            print_header(msg.HEADER_ACTION_REQUIRED)
            print_numbered_list(missing)
            print_blank()
            print_info(msg.MSG_FIX_RERUN)
            raise Exit(1)
        else:
            print_success(msg.MSG_EVERYTHING_OK)
            print_next_steps(["conda activate apxm", "apxm doctor"])
