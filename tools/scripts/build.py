"""Build command for APXM CLI.

Uses sniff.build.BuildSystemDetector for Cargo workspace detection
and sniff.compiler.CompilerDetector to verify rustc availability.
"""

import shutil
import subprocess

from sniff import Typer, Option, Exit

from sniff.build import BuildSystemDetector, BuildSystem
from sniff.compiler import CompilerDetector

from apxm_env import setup_mlir_environment
from sniff import print_error, print_header, print_info, print_step, print_success, print_warning

from . import ensure_conda_env, get_config
from .ci_env import apply_ci_cargo_flags, apply_ci_env, ci_build_hints, detect_ci


def _build_cargo_cmd(package: str, features: list[str] | None = None) -> list[str]:
    """Construct a cargo build command for the given package and features."""
    cmd = ["cargo", "build", "-p", package]
    if features:
        cmd.extend(["--features", ",".join(features)])
    return cmd


def register_commands(app: Typer) -> None:
    """Register build commands on the app."""

    @app.command()
    def build(
        compiler: bool = Option(False, "--compiler", "-c", help="Build compiler only"),
        runtime: bool = Option(False, "--runtime", "-r", help="Build runtime only"),
        release: bool = Option(True, "--release/--debug", help="Build in release mode"),
        clean: bool = Option(False, "--clean", help="Clean build artifacts first"),
        trace: bool = Option(False, "--trace", help="Build with tracing enabled (default)"),
        no_trace: bool = Option(
            False, "--no-trace", help="Build with tracing disabled (zero overhead)"
        ),
    ):
        """Build APXM components.

        By default, builds the full project (compiler + runtime) with tracing enabled.
        Use --compiler or --runtime to build specific components.

        Tracing control:
          --trace     Build with runtime tracing enabled (default)
          --no-trace  Build with tracing compiled out (zero overhead)

        When built with --trace (or default), use 'apxm run --trace <level>' to control
        output at runtime. When built with --no-trace, all tracing is eliminated at
        compile time for maximum performance.
        """
        config = get_config()

        # Detect build system -- verify this is a Cargo workspace
        build_info = BuildSystemDetector().detect_first(config.apxm_dir)
        if build_info is None or build_info.system != BuildSystem.CARGO:
            print_error("No Cargo workspace detected at project root")
            raise Exit(1)
        if not build_info.is_workspace:
            print_warning("Cargo.toml found but is not a workspace")

        # Verify rustc is available
        rustc = CompilerDetector().detect_compiler("rustc")
        if not rustc.found:
            print_error("rustc not found")
            print_info("Activate the conda environment: conda activate apxm")
            print_info("Or install Rust via: curl https://sh.rustup.rs | sh")
            raise Exit(1)
        print_info(f"Cargo workspace: {build_info.root} (edition {build_info.version or 'unknown'})")
        print_info(f"rustc {rustc.version or 'unknown'} ({rustc.target or 'unknown target'})")

        conda_prefix = ensure_conda_env()
        env = setup_mlir_environment(conda_prefix, config.target_dir)

        # Detect CI environment and adapt build settings
        ci = detect_ci()
        hints = ci_build_hints(ci)
        if ci.is_ci:
            env = apply_ci_env(env, hints)
            provider = ci.provider.display_name if ci.provider else "Unknown CI"
            print_info(f"CI detected: {provider}")
            if hints.max_jobs:
                print_info(f"Parallelism capped at {hints.max_jobs} jobs")

        if trace and no_trace:
            print_error("Cannot specify both --trace and --no-trace")
            raise Exit(1)

        if not compiler and not runtime:
            targets = ["full"]
        else:
            targets = []
            if compiler:
                targets.append("compiler")
            if runtime:
                targets.append("runtime")

        features = ["driver", "metrics"]
        if no_trace:
            features.append("no-trace")
            print_header("Building APXM (no-trace)")
            print_info("Tracing disabled - zero overhead build")
        else:
            print_header("Building APXM")
            if trace:
                print_info("Tracing enabled - use --trace flag at runtime")

        if clean:
            print_step("Cleaning build artifacts...")
            build_dir = config.target_dir / "release" / "build"
            if build_dir.exists():
                for pattern in ["apxm-*"]:
                    for p in build_dir.glob(pattern):
                        shutil.rmtree(p, ignore_errors=True)
            print_success("Build artifacts cleaned")

        for target in targets:
            if target == "full":
                print_step("Building full project...")
                cmd = _build_cargo_cmd("apxm-cli", features)
            elif target == "compiler":
                print_step("Building compiler...")
                cmd = _build_cargo_cmd("apxm-compiler")
            elif target == "runtime":
                print_step("Building runtime...")
                runtime_features = ["no-trace"] if no_trace else []
                cmd = _build_cargo_cmd("apxm-runtime", runtime_features or None)

            if release:
                cmd.append("--release")

            cmd = apply_ci_cargo_flags(cmd, hints)

            result = subprocess.run(cmd, cwd=config.apxm_dir, env=env)

            if result.returncode != 0:
                print_error(f"Build failed: {target}")
                raise Exit(1)

            print_success(f"Built: {target}")

        print_success("Build complete!")
        if "full" in targets or "compiler" in targets:
            print_info(f"Binary: {config.compiler_bin}")
        if no_trace:
            print_warning("Note: --trace flag at runtime will have no effect")
