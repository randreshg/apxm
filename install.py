#!/usr/bin/env python3
"""
APXM Bootstrap Installer - No dependencies required.

This script handles first-time installation without requiring sniff.
After installation, use the full 'apxm' CLI for all other commands.

Usage:
    python3 install.py              # Full install
    python3 install.py --check      # Check status only
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def print_step(msg):
    """Print a step message."""
    print(f"\n▶ {msg}")


def print_success(msg):
    """Print a success message."""
    print(f"✓ {msg}")


def print_error(msg):
    """Print an error message."""
    print(f"✗ {msg}", file=sys.stderr)


def print_info(msg):
    """Print an info message."""
    print(f"  {msg}")


def find_conda_cmd():
    """Find mamba or conda command."""
    for cmd in ["mamba", "conda"]:
        if shutil.which(cmd):
            return cmd
    return None


def main():
    """Run the installation."""
    check_only = "--check" in sys.argv

    print("\nAPXM Bootstrap Installer")
    print("=" * 50)

    # Get paths
    apxm_dir = Path(__file__).parent.resolve()
    env_yaml = apxm_dir / "environment.yaml"

    missing = []

    # Step 1: Platform
    print_step("Detecting platform")
    import platform
    print_info(f"{platform.system()} ({platform.machine()})")

    # Step 2: Check conda
    print_step("Checking for conda/mamba")
    conda_cmd = find_conda_cmd()
    if not conda_cmd:
        print_error("Mamba/Conda not found")
        print_info("Install miniforge: https://github.com/conda-forge/miniforge")
        missing.append("mamba/conda")
    else:
        print_success(f"Found: {conda_cmd}")

    # Step 3: Conda environment
    if conda_cmd and not check_only:
        print_step("Creating conda environment")
        if not env_yaml.exists():
            print_error(f"Missing: {env_yaml}")
            missing.append("environment.yaml")
        else:
            result = subprocess.run(
                [conda_cmd, "env", "create", "-f", str(env_yaml), "-n", "apxm"],
                capture_output=True,
            )
            if result.returncode != 0:
                # Try update instead
                result = subprocess.run(
                    [conda_cmd, "env", "update", "-f", str(env_yaml), "-n", "apxm"],
                )
                if result.returncode == 0:
                    print_success("Environment updated")
                else:
                    print_error("Failed to create/update environment")
                    missing.append("conda environment")
            else:
                print_success("Environment created")

    # Step 4: Check for rustup
    print_step("Checking for Rust")
    if not shutil.which("rustup"):
        print_error("Rustup not found")
        if not check_only and "--auto" in sys.argv:
            print_info("Installing rustup...")
            subprocess.run([
                "sh", "-c",
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly"
            ])
        else:
            print_info("Install: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
            missing.append("rustup")
    else:
        print_success("Rustup found")

    # Step 5: Build
    if not check_only and not missing:
        print_step("Building APXM")

        # Check if conda env exists
        result = subprocess.run(
            [conda_cmd, "env", "list"],
            capture_output=True,
            text=True,
        )
        if "apxm" not in result.stdout:
            print_error("Conda environment 'apxm' not found")
            print_info("Run without --check to create it")
            sys.exit(1)

        # Get conda prefix
        result = subprocess.run(
            [conda_cmd, "run", "-n", "apxm", "python", "-c", "import sys; print(sys.prefix)"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print_error("Failed to get conda prefix")
            sys.exit(1)

        conda_prefix = result.stdout.strip()

        # Set up environment
        env = os.environ.copy()
        env["MLIR_DIR"] = str(Path(conda_prefix) / "lib" / "cmake" / "mlir")
        env["LLVM_DIR"] = str(Path(conda_prefix) / "lib" / "cmake" / "llvm")

        # Build
        result = subprocess.run(
            [conda_cmd, "run", "-n", "apxm", "cargo", "build", "-p", "apxm-cli", "--features", "driver,metrics", "--release"],
            cwd=apxm_dir,
            env=env,
        )

        if result.returncode == 0:
            print_success("Build complete")

            # Install binary
            bin_dir = apxm_dir / "bin"
            bin_dir.mkdir(exist_ok=True)

            source = apxm_dir / "target" / "release" / "apxm"
            target = bin_dir / "apxm"

            if source.exists():
                try:
                    if target.exists():
                        target.unlink()
                    target.symlink_to(source.resolve())
                    print_success(f"Installed: {target}")
                except OSError:
                    shutil.copy2(source, target)
                    target.chmod(0o755)
                    print_success(f"Installed: {target}")

                print_info("\nNext steps:")
                print_info(f"1. Add to PATH: export PATH=\"{bin_dir}:$PATH\"")
                print_info(f"2. Add to shell config: echo 'export PATH=\"{bin_dir}:$PATH\"' >> ~/.bashrc")
                print_info("3. Restart shell or: source ~/.bashrc")
                print_info("4. Run: apxm doctor")
            else:
                print_error("Binary not found after build")
        else:
            print_error("Build failed")
            missing.append("build")

    # Summary
    if missing:
        print("\n❌ Installation incomplete:")
        for item in missing:
            print(f"  - {item}")
        sys.exit(1)
    elif check_only:
        print("\n✓ All dependencies available")
    else:
        print("\n✓ Installation complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        sys.exit(1)
