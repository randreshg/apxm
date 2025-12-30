#!/usr/bin/env python3
"""
APXM CLI - Python driver for APXM compiler and runtime.

Automatically handles conda environment detection, MLIR environment setup,
and provides convenient commands for building, running, and testing.

Usage:
    python tools/apxm_cli.py doctor
    python tools/apxm_cli.py compiler build
    python tools/apxm_cli.py compiler run examples/hello_world.ais
    python tools/apxm_cli.py workloads check
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from apxm_env import ApxmConfig, get_conda_prefix, setup_mlir_environment
from apxm_styles import (
    Colors,
    console,
    print_error,
    print_header,
    print_info,
    print_step,
    print_success,
    print_warning,
)

# CLI Apps
app = typer.Typer(
    name="apxm-cli",
    help="APXM CLI - Compiler and runtime driver",
    no_args_is_help=True,
)
compiler_app = typer.Typer(help="Compiler operations")
workloads_app = typer.Typer(help="Workload validation and benchmarks")

app.add_typer(compiler_app, name="compiler")
app.add_typer(workloads_app, name="workloads")


# Configuration
_config: Optional[ApxmConfig] = None


def get_config() -> ApxmConfig:
    """Get or create the global configuration."""
    global _config
    if _config is None:
        _config = ApxmConfig.detect()
    return _config


def ensure_conda_env() -> Path:
    """Ensure conda environment is available, exit if not."""
    conda_prefix = get_conda_prefix()
    if not conda_prefix:
        print_error("Conda environment 'apxm' not found!")
        print_info("Create it with: cargo run -p apxm-cli -- install")
        raise typer.Exit(1)
    return conda_prefix


def compile_ais(
    file: Path, env: dict[str, str], config: ApxmConfig, output: Optional[Path] = None
) -> subprocess.CompletedProcess:
    """Compile an AIS file."""
    cmd = [str(config.compiler_bin), "compile", str(file)]
    if output:
        cmd.extend(["-o", str(output)])
    else:
        cmd.extend(["-o", "/dev/null"])

    return subprocess.run(cmd, env=env, capture_output=True, text=True)


# Doctor Command
@app.command()
def doctor():
    """Check environment status and dependencies."""
    config = get_config()

    print_header("APXM Environment Check")

    # Check APXM directory
    if config.apxm_dir.exists():
        print_success(f"APXM directory: {config.apxm_dir}")
    else:
        print_error(f"APXM directory not found: {config.apxm_dir}")

    # Check Rust toolchain
    try:
        result = subprocess.run(
            ["rustc", "--version"], capture_output=True, text=True, check=True
        )
        print_success(f"Rust: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("Rust toolchain not found")

    # Check Cargo
    try:
        result = subprocess.run(
            ["cargo", "--version"], capture_output=True, text=True, check=True
        )
        print_success(f"Cargo: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("Cargo not found")

    # Check conda environment
    conda_prefix = get_conda_prefix()
    if conda_prefix:
        print_success(f"Conda environment: {conda_prefix}")

        # Check MLIR
        mlir_dir = conda_prefix / "lib" / "cmake" / "mlir"
        if mlir_dir.exists():
            print_success(f"MLIR: {mlir_dir}")
        else:
            print_error(f"MLIR not found at {mlir_dir}")

        # Check LLVM
        llvm_dir = conda_prefix / "lib" / "cmake" / "llvm"
        if llvm_dir.exists():
            print_success(f"LLVM: {llvm_dir}")
        else:
            print_error(f"LLVM not found at {llvm_dir}")
    else:
        print_error("Conda environment 'apxm' not found")
        print_info("  Create with: cargo run -p apxm-cli -- install")

    # Check compiler binary
    if config.compiler_bin.exists():
        print_success(f"Compiler binary: {config.compiler_bin}")
    else:
        print_warning("Compiler not built")
        print_info("  Build with: python tools/apxm_cli.py compiler build")

    # Check workloads directory
    if config.workloads_dir.exists():
        workflows = list(config.workloads_dir.glob("*/workflow.ais"))
        print_success(f"Workloads: {len(workflows)} workflows found")
    else:
        print_warning(f"Workloads directory not found: {config.workloads_dir}")

    console.print()


# Compiler Commands
@compiler_app.command("build")
def compiler_build(
    release: bool = typer.Option(True, "--release/--debug", help="Build in release mode"),
    clean: bool = typer.Option(False, "--clean", "-c", help="Clean build artifacts first"),
):
    """Build the APXM compiler."""
    config = get_config()
    conda_prefix = ensure_conda_env()
    env = setup_mlir_environment(conda_prefix, config.target_dir)

    print_header("Building APXM Compiler")

    if clean:
        print_step("Cleaning build artifacts...")
        build_dir = config.target_dir / "release" / "build"
        if build_dir.exists():
            for pattern in ["apxm-compiler-*", "apxm-driver-*"]:
                for p in build_dir.glob(pattern):
                    shutil.rmtree(p, ignore_errors=True)
        print_success("Build artifacts cleaned")

    print_step("Building compiler (this may take a while)...")

    cmd = ["cargo", "build", "-p", "apxm-cli", "--features", "driver"]
    if release:
        cmd.append("--release")

    result = subprocess.run(cmd, cwd=config.apxm_dir, env=env)

    if result.returncode == 0:
        print_success("Compiler built successfully!")
        print_info(f"Binary: {config.compiler_bin}")
    else:
        print_error("Build failed!")
        raise typer.Exit(1)


@compiler_app.command("run")
def compiler_run(
    file: Path = typer.Argument(..., help="AIS file to compile and run"),
    optimization: str = typer.Option("O1", "-O", help="Optimization level (O0/O1/O2)"),
    emit_diagnostics: Optional[Path] = typer.Option(
        None, "--emit-diagnostics", help="Emit diagnostics to file"
    ),
    cargo: bool = typer.Option(
        False, "--cargo", help="Use cargo run instead of pre-built binary (slower, but auto-rebuilds)"
    ),
):
    """Compile and run an AIS file."""
    config = get_config()
    conda_prefix = ensure_conda_env()
    env = setup_mlir_environment(conda_prefix, config.target_dir)

    if not file.exists():
        print_error(f"File not found: {file}")
        raise typer.Exit(1)

    print_header(f"Running {file.name}")

    if cargo:
        # Use cargo run (auto-rebuilds if needed)
        cmd = [
            "cargo", "run", "-p", "apxm-cli", "--features", "driver", "--release",
            "--", "run", str(file), f"-{optimization}"
        ]
        if emit_diagnostics:
            cmd.extend(["--emit-diagnostics", str(emit_diagnostics)])
    else:
        # Use pre-built binary (faster)
        if not config.compiler_bin.exists():
            print_error("Compiler not built!")
            print_info("Run: python tools/apxm_cli.py compiler build")
            print_info("Or use --cargo to auto-build")
            raise typer.Exit(1)
        cmd = [str(config.compiler_bin), "run", str(file), f"-{optimization}"]
        if emit_diagnostics:
            cmd.extend(["--emit-diagnostics", str(emit_diagnostics)])

    result = subprocess.run(cmd, cwd=config.apxm_dir, env=env)
    raise typer.Exit(result.returncode)


@compiler_app.command("compile")
def compiler_compile(
    file: Path = typer.Argument(..., help="AIS file to compile"),
    output: Path = typer.Option(..., "-o", "--output", help="Output artifact path"),
    emit_mlir: bool = typer.Option(False, "--emit-mlir", help="Emit MLIR output"),
    optimization: str = typer.Option("O1", "-O", help="Optimization level (O0/O1/O2)"),
    cargo: bool = typer.Option(
        False, "--cargo", help="Use cargo run instead of pre-built binary (slower, but auto-rebuilds)"
    ),
):
    """Compile an AIS file to an artifact."""
    config = get_config()
    conda_prefix = ensure_conda_env()
    env = setup_mlir_environment(conda_prefix, config.target_dir)

    if not file.exists():
        print_error(f"File not found: {file}")
        raise typer.Exit(1)

    print_step(f"Compiling {file.name}...")

    if cargo:
        # Use cargo run (auto-rebuilds if needed)
        cmd = [
            "cargo", "run", "-p", "apxm-cli", "--features", "driver", "--release",
            "--", "compile", str(file), "-o", str(output), f"-{optimization}"
        ]
        if emit_mlir:
            cmd.append("--emit-mlir")
    else:
        # Use pre-built binary (faster)
        if not config.compiler_bin.exists():
            print_error("Compiler not built!")
            print_info("Run: python tools/apxm_cli.py compiler build")
            print_info("Or use --cargo to auto-build")
            raise typer.Exit(1)
        cmd = [str(config.compiler_bin), "compile", str(file), "-o", str(output), f"-{optimization}"]
        if emit_mlir:
            cmd.append("--emit-mlir")

    result = subprocess.run(cmd, cwd=config.apxm_dir, env=env)

    if result.returncode == 0:
        print_success(f"Compiled to {output}")
    else:
        print_error("Compilation failed!")

    raise typer.Exit(result.returncode)


# Workload Commands
@workloads_app.command("list")
def workloads_list():
    """List available benchmark workloads."""
    config = get_config()

    print_header("Available Workloads")

    if not config.workloads_dir.exists():
        print_error(f"Workloads directory not found: {config.workloads_dir}")
        raise typer.Exit(1)

    workflows = sorted(config.workloads_dir.glob("*/workflow.ais"))
    disabled = sorted(config.workloads_dir.glob("*/workflow.ais.disabled"))

    for workflow in workflows:
        name = workflow.parent.name
        readme = workflow.parent / "README.md"
        desc = ""
        if readme.exists():
            with open(readme) as f:
                first_line = f.readline().strip()
                if first_line.startswith("#"):
                    desc = first_line.lstrip("# ")
        console.print(f"  [{Colors.PASS}]\u2713[/{Colors.PASS}] {name:30} {desc}")

    for workflow in disabled:
        name = workflow.parent.name
        console.print(f"  [{Colors.DIM}]\u25cb[/{Colors.DIM}] {name:30} [dim](disabled)[/dim]")

    console.print()
    console.print(f"  Total: {len(workflows)} active, {len(disabled)} disabled")


@workloads_app.command("check")
def workloads_check(
    quick: bool = typer.Option(False, "--quick", "-q", help="Quick mode (skip slow workloads)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show compilation output"),
):
    """Compile all benchmark workloads to verify they work."""
    _ = quick  # Reserved for future use (skip slow benchmarks)
    config = get_config()
    conda_prefix = ensure_conda_env()
    env = setup_mlir_environment(conda_prefix, config.target_dir)

    if not config.compiler_bin.exists():
        print_error("Compiler not built!")
        print_info("Run: python tools/apxm_cli.py compiler build")
        raise typer.Exit(1)

    workflows = sorted(config.workloads_dir.glob("*/workflow.ais"))

    print_header(f"Checking {len(workflows)} Workloads")

    passed, failed, skipped = 0, 0, 0
    failed_names = []

    for i, workflow in enumerate(workflows, 1):
        name = workflow.parent.name
        print_step(f"[{i}/{len(workflows)}] {name}...")

        result = compile_ais(workflow, env, config)

        if result.returncode == 0:
            print_success(name)
            passed += 1
        else:
            print_error(name)
            failed += 1
            failed_names.append(name)
            if verbose:
                console.print(f"    [dim]{result.stderr[:500]}[/dim]")

    # Summary
    console.print()
    print_header("Summary")

    if passed > 0:
        console.print(f"  [{Colors.PASS}]\u2713 {passed}[/{Colors.PASS}] passed")
    if failed > 0:
        console.print(f"  [{Colors.FAIL}]\u2717 {failed}[/{Colors.FAIL}] failed")
        for name in failed_names:
            console.print(f"    - {name}")
    if skipped > 0:
        console.print(f"  [{Colors.DIM}]\u25cb {skipped}[/{Colors.DIM}] skipped")

    console.print()

    if failed > 0:
        raise typer.Exit(1)


@workloads_app.command("run")
def workloads_run(
    name: str = typer.Argument(..., help="Workload name (e.g., 10_multi_agent)"),
):
    """Run a specific workload benchmark."""
    config = get_config()
    conda_prefix = ensure_conda_env()
    env = setup_mlir_environment(conda_prefix, config.target_dir)

    workload_dir = config.workloads_dir / name
    if not workload_dir.exists():
        print_error(f"Workload not found: {name}")
        print_info("Run 'workloads list' to see available workloads")
        raise typer.Exit(1)

    run_script = workload_dir / "run.py"
    if not run_script.exists():
        print_error(f"No run.py found in {name}")
        raise typer.Exit(1)

    print_header(f"Running Workload: {name}")

    result = subprocess.run(
        [sys.executable, str(run_script)],
        cwd=workload_dir,
        env=env,
    )
    raise typer.Exit(result.returncode)


@workloads_app.command("benchmark")
def workloads_benchmark(
    name: Optional[str] = typer.Argument(None, help="Workload name (e.g., 2_chain_fusion)"),
    all_workloads: bool = typer.Option(False, "--all", "-a", help="Run all workloads"),
    iterations: int = typer.Option(10, "-n", "--iterations", help="Benchmark iterations"),
    warmup: int = typer.Option(3, "-w", "--warmup", help="Warmup iterations (not counted)"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON format"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Save results to file"),
):
    """Run benchmark(s) with proper environment setup.

    Examples:
        workloads benchmark 2_chain_fusion
        workloads benchmark --all --json -o results.json
    """
    config = get_config()
    conda_prefix = ensure_conda_env()
    env = setup_mlir_environment(conda_prefix, config.target_dir)

    if not config.compiler_bin.exists():
        print_error("Compiler not built!")
        print_info("Run: python tools/apxm_cli.py compiler build")
        raise typer.Exit(1)

    # Import the runner module
    runner_path = config.workloads_dir / "runner.py"
    if not runner_path.exists():
        print_error(f"Runner not found: {runner_path}")
        raise typer.Exit(1)

    # Build command for runner
    cmd = [sys.executable, str(runner_path)]

    if all_workloads:
        pass  # Run all workloads (default)
    elif name:
        # Map name to workload number if needed
        workload_num = None
        for workflow in config.workloads_dir.glob("*/workflow.ais"):
            dir_name = workflow.parent.name
            if dir_name == name or dir_name.endswith(f"_{name}"):
                # Extract number from directory name like "2_chain_fusion"
                parts = dir_name.split("_")
                if parts[0].isdigit():
                    workload_num = int(parts[0])
                break

        if workload_num:
            cmd.extend(["--workload", str(workload_num)])
        else:
            print_error(f"Workload not found: {name}")
            print_info("Run 'workloads list' to see available workloads")
            raise typer.Exit(1)
    else:
        print_error("Specify a workload name or use --all")
        raise typer.Exit(1)

    cmd.extend(["--iterations", str(iterations), "--warmup", str(warmup)])

    if json_output:
        cmd.append("--json")

    if output:
        cmd.extend(["--output", str(output)])

    if not json_output and not output:
        print_header("Running Benchmark")
        if name:
            print_info(f"Workload: {name}")
        else:
            print_info("Running all workloads")
        print_info(f"Iterations: {iterations}, Warmup: {warmup}")
        console.print()

    result = subprocess.run(cmd, cwd=config.workloads_dir, env=env)

    if result.returncode == 0 and output and not json_output:
        print_success(f"Results saved to: {output}")

    raise typer.Exit(result.returncode)


if __name__ == "__main__":
    app()
