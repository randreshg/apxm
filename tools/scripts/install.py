"""Enhanced install command for APXM CLI."""

import shutil
import subprocess
from pathlib import Path

import typer

from apxm_env import get_conda_prefix, setup_mlir_environment
from apxm_styles import (
    print_blank,
    print_error,
    print_header,
    print_info,
    print_next_steps,
    print_numbered_list,
    print_step,
    print_success,
    print_warning,
)

from tools.scripts import get_config

from .config import get_platform_config
from .deps import check_all


def register_commands(app: typer.Typer) -> None:
    """Register install command on the app."""

    @app.command()
    def install(
        check: bool = typer.Option(False, "--check", help="Dry-run: report status without changes"),
        skip_deps: bool = typer.Option(False, "--skip-deps", help="Skip dependency checks"),
        skip_build: bool = typer.Option(False, "--skip-build", help="Skip build step"),
        auto: bool = typer.Option(
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
        platform = get_platform_config()
        missing: list[str] = []

        print_header("APXM Install")

        # ── Stage 1 ───────────────────────────────────────────────
        print_step("Stage 1: Platform detection")
        print_info(f"OS: {platform.os_name} ({platform.arch})")
        if platform.distro:
            print_info(f"Distro: {platform.distro} {platform.distro_version or ''}")
        if platform.is_wsl:
            print_info("WSL detected")
        if platform.pkg_manager:
            print_info(f"Package manager: {platform.pkg_manager}")
        print_blank()

        # ── Stage 2 ───────────────────────────────────────────────
        # Mamba/Conda and Rust are checked in detail by Stages 3 & 4,
        # so Stage 2 only reports their status without adding to missing.
        stage3_4_names = {"Mamba/Conda", "Rust (nightly)", "Cargo"}
        if not skip_deps:
            print_step("Stage 2: Checking dependencies")
            results = check_all()
            for r in results:
                if r.found:
                    version_str = f" ({r.version})" if r.version else ""
                    if r.meets_minimum:
                        print_success(f"{r.dep.name}{version_str}")
                    else:
                        print_warning(
                            f"{r.dep.name}{version_str} -- "
                            f"needs >= {r.dep.min_version}"
                        )
                        if r.dep.name not in stage3_4_names:
                            missing.append(f"{r.dep.name} (upgrade to >= {r.dep.min_version})")
                else:
                    if r.dep.required:
                        print_error(f"{r.dep.name} -- not found")
                        if r.dep.name not in stage3_4_names:
                            missing.append(r.dep.name)
                    else:
                        print_warning(f"{r.dep.name} -- not found (optional)")
            print_blank()

        # ── Stage 3 ───────────────────────────────────────────────
        print_step("Stage 3: Conda environment")
        conda_cmd = platform.conda_cmd
        if not conda_cmd:
            print_error("Mamba/Conda -- not found")
            missing.append("Mamba/Conda")
            print_blank()
        else:
            print_success(f"{conda_cmd}: found")

            env_yaml = config.apxm_dir / "environment.yaml"
            if not env_yaml.exists():
                print_error(f"environment.yaml not found: {env_yaml}")
                missing.append("environment.yaml")
            elif check:
                conda_prefix = get_conda_prefix()
                if conda_prefix:
                    print_success(f"Conda env 'apxm': {conda_prefix}")
                else:
                    print_warning("Conda env 'apxm': not found (will be created)")
                    missing.append("Conda env 'apxm' (run install without --check)")
            else:
                print_info("Creating/updating conda environment...")
                result = subprocess.run(
                    [conda_cmd, "env", "create", "-f", str(env_yaml)],
                    capture_output=False,
                )
                if result.returncode != 0:
                    result = subprocess.run(
                        [conda_cmd, "env", "update", "-f", str(env_yaml), "-n", "apxm"],
                        capture_output=False,
                    )
                    if result.returncode != 0:
                        print_error(f"{conda_cmd} env create/update failed")
                        missing.append("Conda env (create/update failed)")
                    else:
                        print_success("Conda environment updated")
                else:
                    print_success("Conda environment created")
            print_blank()

        # ── Stage 4 ───────────────────────────────────────────────
        print_step("Stage 4: Rust toolchain")
        has_rustup = shutil.which("rustup") is not None
        if has_rustup:
            print_success("rustup: found")
            if not check:
                try:
                    result = subprocess.run(
                        ["rustup", "show"],
                        capture_output=True, text=True, timeout=30,
                    )
                    if "nightly" in result.stdout:
                        print_success("Rust nightly: installed")
                    else:
                        print_warning("Rust nightly not found, installing...")
                        subprocess.run(
                            ["rustup", "toolchain", "install", "nightly"],
                            capture_output=False, timeout=300,
                        )
                except (subprocess.TimeoutExpired, OSError) as e:
                    print_warning(f"Could not check Rust toolchain: {e}")
            else:
                try:
                    result = subprocess.run(
                        ["rustc", "--version"],
                        capture_output=True, text=True, timeout=10,
                    )
                    version = result.stdout.strip()
                    if "nightly" in version:
                        print_success(f"Rust: {version}")
                    else:
                        print_warning(f"Rust: {version} (nightly recommended)")
                        missing.append("Rust nightly (rustup toolchain install nightly)")
                except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                    print_warning("Could not determine Rust version")
        else:
            print_error("rustup -- not found")
            missing.append("rustup")
            if not check and auto:
                print_info("Installing rustup...")
                try:
                    subprocess.run(
                        [
                            "sh", "-c",
                            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly",
                        ],
                        capture_output=False, timeout=300,
                    )
                except (subprocess.TimeoutExpired, OSError) as e:
                    print_error(f"Failed to install rustup: {e}")
        print_blank()

        # ── Stage 5 ───────────────────────────────────────────────
        if not skip_build:
            print_step("Stage 5: Build")
            has_cargo = shutil.which("cargo") is not None
            conda_prefix = get_conda_prefix()

            if check:
                if config.compiler_bin.exists():
                    print_success(f"Binary: {config.compiler_bin}")
                else:
                    print_warning("Compiler not built yet")
                    missing.append("Build (conda activate apxm && apxm build)")
            elif not has_cargo:
                print_warning("Cargo not available -- skipping build")
                missing.append("Build (install Rust first)")
            elif not conda_prefix:
                print_warning("Conda env not activated -- skipping build")
                missing.append("Build (conda activate apxm && apxm build)")
            else:
                env = setup_mlir_environment(conda_prefix, config.target_dir)
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
                    print_success("Build complete!")
                else:
                    print_error("Build failed!")
                    missing.append("Build (fix errors and retry: apxm build)")
        print_blank()

        # ── Stage 6: Summary ──────────────────────────────────────
        if missing:
            print_header("Action Required")
            print_numbered_list(missing)
            print_blank()
            print_info("Fix the above, then re-run: apxm install")
            raise typer.Exit(1)
        else:
            print_success("Everything looks good!")
            print_next_steps(["conda activate apxm", "apxm doctor"])
