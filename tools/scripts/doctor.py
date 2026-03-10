"""Enhanced doctor command for APXM CLI."""

import shutil
import subprocess

import typer

from apxm_env import get_conda_prefix
from apxm_styles import (
    print_blank,
    print_dim,
    print_error,
    print_header,
    print_info,
    print_section,
    print_success,
    print_warning,
)

from tools.scripts import get_config

from .deps import check_all


def register_commands(app: typer.Typer) -> None:
    """Register doctor command on the app."""

    @app.command()
    def doctor():
        """Check environment status and dependencies."""
        config = get_config()
        errors = 0

        print_header("APXM Environment Check")

        # APXM directory
        if config.apxm_dir.exists():
            print_success(f"APXM directory: {config.apxm_dir}")
        else:
            print_error(f"APXM directory not found: {config.apxm_dir}")
            errors += 1

        # Dependency checks
        print_section("Dependencies:")
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
                    if r.dep.required:
                        errors += 1
            else:
                if r.dep.required:
                    print_error(f"{r.dep.name} -- not found")
                    errors += 1
                else:
                    print_warning(f"{r.dep.name} -- not found (optional)")

        # Rust toolchain details
        print_section("Rust Toolchain:")
        try:
            result = subprocess.run(
                ["rustc", "--version"], capture_output=True, text=True, check=True
            )
            version = result.stdout.strip()
            if "nightly" in version:
                print_success(f"Rust: {version}")
            else:
                print_warning(f"Rust: {version} (nightly recommended)")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print_error("Rust toolchain not found")
            errors += 1

        # Conda environment
        print_section("Conda Environment:")
        has_conda = shutil.which("mamba") or shutil.which("conda")
        if not has_conda:
            print_error("Conda/Mamba is not installed")
            print_info("Install with: curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh | bash")
            errors += 1
        else:
            conda_prefix = get_conda_prefix()
            if conda_prefix:
                print_success(f"Conda environment 'apxm': {conda_prefix}")

                # Check MLIR
                mlir_dir = conda_prefix / "lib" / "cmake" / "mlir"
                if mlir_dir.exists():
                    print_success(f"MLIR: {mlir_dir}")
                    # Check MLIR version
                    mlir_tblgen = conda_prefix / "bin" / "mlir-tblgen"
                    if mlir_tblgen.exists():
                        try:
                            result = subprocess.run(
                                [str(mlir_tblgen), "--version"],
                                capture_output=True, text=True,
                            )
                            output = result.stdout + result.stderr
                            if "21" in output:
                                print_success("MLIR version: 21.x")
                            else:
                                print_warning(f"MLIR version: {output.strip()} (expected 21.x)")
                        except FileNotFoundError:
                            pass
                else:
                    print_error(f"MLIR not found at {mlir_dir}")
                    errors += 1

                # Check LLVM
                llvm_dir = conda_prefix / "lib" / "cmake" / "llvm"
                if llvm_dir.exists():
                    print_success(f"LLVM: {llvm_dir}")
                else:
                    print_error(f"LLVM not found at {llvm_dir}")
                    errors += 1
            else:
                print_warning("Conda env 'apxm' not found")
                print_info("Create with: apxm install")
                errors += 1

        # Build status
        print_section("Build Status:")
        if config.compiler_bin.exists():
            print_success(f"Compiler binary: {config.compiler_bin}")
        else:
            print_warning("Compiler not built")
            print_info("Build with: apxm build")

        # Registered credentials
        print_section("Registered Credentials:")
        try:
            if config.compiler_bin.exists():
                result = subprocess.run(
                    [str(config.compiler_bin), "register", "list"],
                    capture_output=True, text=True,
                    env=config.get_mlir_env(),
                )
                output = result.stdout.strip()
                if result.returncode == 0 and output and "No credentials" not in output:
                    for line in output.split("\n"):
                        print_dim(line)
                else:
                    print_info("No credentials registered")
                    print_info("Register with: apxm register add <name> --provider <provider>")
            else:
                print_info("Build CLI first to check credentials")
        except FileNotFoundError:
            print_info("No credentials registered")

        print_blank()

        if errors > 0:
            raise typer.Exit(1)
