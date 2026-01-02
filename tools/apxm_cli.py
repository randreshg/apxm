#!/usr/bin/env python3
"""
APXM CLI - Python driver for APXM compiler and runtime.

Automatically handles conda environment detection, MLIR environment setup,
and provides convenient commands for building, running, and testing.

Usage:
    apxm doctor                     # Check environment status
    apxm build                      # Build compiler and runtime
    apxm execute workflow.ais       # Compile and execute an AIS file
    apxm compile workflow.ais -o out.apxmobj  # Compile to artifact
    apxm run out.apxmobj            # Run pre-compiled artifact
    apxm workloads list             # List available workloads
    apxm workloads check            # Verify all workloads compile
"""

import os
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


# Build Command
@app.command()
def build(
    compiler: bool = typer.Option(False, "--compiler", "-c", help="Build compiler only"),
    runtime: bool = typer.Option(False, "--runtime", "-r", help="Build runtime only"),
    release: bool = typer.Option(True, "--release/--debug", help="Build in release mode"),
    clean: bool = typer.Option(False, "--clean", help="Clean build artifacts first"),
    trace: bool = typer.Option(False, "--trace", help="Build with tracing enabled (default)"),
    no_trace: bool = typer.Option(False, "--no-trace", help="Build with tracing disabled (zero overhead)"),
):
    """Build APXM components.

    By default, builds the full project (compiler + runtime) with tracing enabled.
    Use --compiler or --runtime to build specific components.

    Tracing control:
      --trace     Build with runtime tracing enabled (default)
      --no-trace  Build with tracing compiled out (zero overhead for benchmarks)

    When built with --trace (or default), use 'apxm run --trace <level>' to control
    output at runtime. When built with --no-trace, all tracing is eliminated at
    compile time for maximum performance.
    """
    config = get_config()
    conda_prefix = ensure_conda_env()
    env = setup_mlir_environment(conda_prefix, config.target_dir)

    # Handle conflicting options
    if trace and no_trace:
        print_error("Cannot specify both --trace and --no-trace")
        raise typer.Exit(1)

    # Determine what to build
    if not compiler and not runtime:
        # Build everything
        targets = ["full"]
    else:
        targets = []
        if compiler:
            targets.append("compiler")
        if runtime:
            targets.append("runtime")

    # Build feature set (always include metrics for proper observability)
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
            cmd = ["cargo", "build", "-p", "apxm-cli", "--features", ",".join(features)]
        elif target == "compiler":
            print_step("Building compiler...")
            cmd = ["cargo", "build", "-p", "apxm-compiler"]
        elif target == "runtime":
            print_step("Building runtime...")
            runtime_features = []
            if no_trace:
                runtime_features.append("no-trace")
            if runtime_features:
                cmd = ["cargo", "build", "-p", "apxm-runtime", "--features", ",".join(runtime_features)]
            else:
                cmd = ["cargo", "build", "-p", "apxm-runtime"]

        if release:
            cmd.append("--release")

        result = subprocess.run(cmd, cwd=config.apxm_dir, env=env)

        if result.returncode != 0:
            print_error(f"Build failed: {target}")
            raise typer.Exit(1)

        print_success(f"Built: {target}")

    print_success("Build complete!")
    if "full" in targets or "compiler" in targets:
        print_info(f"Binary: {config.compiler_bin}")
    if no_trace:
        print_warning("Note: --trace flag at runtime will have no effect")


# Execute Command (compile + run .ais files)
@app.command(context_settings={"allow_interspersed_args": False})
def execute(
    file: Path = typer.Argument(..., help="AIS file to compile and execute"),
    args: Optional[list[str]] = typer.Argument(None, help="Arguments for entry flow"),
    opt_level: int = typer.Option(1, "-O", "--opt-level", help="Optimization level (0-3)"),
    trace: Optional[str] = typer.Option(
        None, "--trace", help="Enable tracing (levels: trace, debug, info, warn, error)"
    ),
    emit_metrics: Optional[Path] = typer.Option(
        None, "--emit-metrics", help="Emit runtime metrics to JSON file"
    ),
    cargo: bool = typer.Option(
        False, "--cargo", help="Use cargo run instead of pre-built binary (slower, but auto-rebuilds)"
    ),
):
    """Compile and execute an AIS file.

    Options must come BEFORE the file path. Arguments after file are passed to entry flow:
        apxm execute [options] workflow.ais [args...]

    Examples:
        apxm execute workflow.ais "quantum computing"
        apxm execute --emit-metrics metrics.json workflow.ais "topic"
        apxm execute -O2 --trace debug workflow.ais "input"
    """
    config = get_config()
    conda_prefix = ensure_conda_env()
    env = setup_mlir_environment(conda_prefix, config.target_dir)

    # Resolve to absolute paths (relative to current working directory)
    cwd = Path.cwd()
    file = file.resolve()
    if emit_metrics and not emit_metrics.is_absolute():
        emit_metrics = (cwd / emit_metrics).resolve()

    if not file.exists():
        print_error(f"File not found: {file}")
        raise typer.Exit(1)

    print_header(f"Executing {file.name}")

    if cargo:
        cmd = [
            "cargo", "run", "-p", "apxm-cli", "--features", "driver,metrics", "--release",
            "--", "execute", f"-O{opt_level}"
        ]
        if trace:
            cmd.extend(["--trace", trace])
        if emit_metrics:
            cmd.extend(["--emit-metrics", str(emit_metrics)])
        cmd.append(str(file))
        if args:
            cmd.extend(args)
    else:
        if not config.compiler_bin.exists():
            print_error("Compiler not built!")
            print_info("Run: apxm build")
            print_info("Or use --cargo to auto-build")
            raise typer.Exit(1)
        cmd = [str(config.compiler_bin), "execute", f"-O{opt_level}"]
        if trace:
            cmd.extend(["--trace", trace])
        if emit_metrics:
            cmd.extend(["--emit-metrics", str(emit_metrics)])
        cmd.append(str(file))
        if args:
            cmd.extend(args)

    result = subprocess.run(cmd, cwd=config.apxm_dir, env=env)
    raise typer.Exit(result.returncode)


# Compile Command (compile .ais to .apxmobj)
@app.command()
def compile(
    file: Path = typer.Argument(..., help="AIS file to compile"),
    output: Path = typer.Option(..., "-o", "--output", help="Output artifact path"),
    emit_diagnostics: Optional[Path] = typer.Option(
        None, "--emit-diagnostics", help="Emit diagnostics JSON file"
    ),
    opt_level: int = typer.Option(1, "-O", "--opt-level", help="Optimization level (0-3)"),
    cargo: bool = typer.Option(
        False, "--cargo", help="Use cargo run instead of pre-built binary (slower, but auto-rebuilds)"
    ),
):
    """Compile an AIS file to an artifact (.apxmobj)."""
    config = get_config()
    conda_prefix = ensure_conda_env()
    env = setup_mlir_environment(conda_prefix, config.target_dir)

    # Resolve to absolute paths (relative to current working directory)
    cwd = Path.cwd()
    file = file.resolve()
    output = (cwd / output).resolve() if not output.is_absolute() else output
    if emit_diagnostics:
        emit_diagnostics = (cwd / emit_diagnostics).resolve() if not emit_diagnostics.is_absolute() else emit_diagnostics

    if not file.exists():
        print_error(f"File not found: {file}")
        raise typer.Exit(1)

    print_step(f"Compiling {file.name}...")

    if cargo:
        cmd = [
            "cargo", "run", "-p", "apxm-cli", "--features", "driver,metrics", "--release",
            "--", "compile", str(file), "-o", str(output), f"-O{opt_level}"
        ]
        if emit_diagnostics:
            cmd.extend(["--emit-diagnostics", str(emit_diagnostics)])
    else:
        if not config.compiler_bin.exists():
            print_error("Compiler not built!")
            print_info("Run: apxm build")
            print_info("Or use --cargo to auto-build")
            raise typer.Exit(1)
        cmd = [str(config.compiler_bin), "compile", str(file), "-o", str(output), f"-O{opt_level}"]
        if emit_diagnostics:
            cmd.extend(["--emit-diagnostics", str(emit_diagnostics)])

    result = subprocess.run(cmd, cwd=config.apxm_dir, env=env)

    if result.returncode == 0:
        print_success(f"Compiled to {output}")
    else:
        print_error("Compilation failed!")

    raise typer.Exit(result.returncode)


# Run Command (run pre-compiled .apxmobj artifacts)
@app.command(context_settings={"allow_interspersed_args": False})
def run(
    file: Path = typer.Argument(..., help="Compiled artifact (.apxmobj) to run"),
    args: Optional[list[str]] = typer.Argument(None, help="Arguments for entry flow"),
    trace: Optional[str] = typer.Option(
        None, "--trace", help="Enable tracing (levels: trace, debug, info, warn, error)"
    ),
    emit_metrics: Optional[Path] = typer.Option(
        None, "--emit-metrics", help="Emit runtime metrics to JSON file"
    ),
):
    """Run a pre-compiled artifact (.apxmobj).

    Use 'apxm execute <file.ais>' to compile and run source files.
    """
    config = get_config()
    conda_prefix = ensure_conda_env()
    env = setup_mlir_environment(conda_prefix, config.target_dir)

    # Resolve to absolute paths (relative to current working directory)
    cwd = Path.cwd()
    file = file.resolve()
    if emit_metrics:
        emit_metrics = (cwd / emit_metrics).resolve() if not emit_metrics.is_absolute() else emit_metrics

    if not file.exists():
        print_error(f"File not found: {file}")
        raise typer.Exit(1)

    # Validate file extension
    if file.suffix != ".apxmobj":
        print_error(f"Expected .apxmobj artifact file, got: {file.suffix or 'no extension'}")
        print_info("Use 'apxm execute <file.ais>' to compile and run source files.")
        raise typer.Exit(1)

    if not config.compiler_bin.exists():
        print_error("Runtime not built!")
        print_info("Run: apxm build")
        raise typer.Exit(1)

    print_header(f"Running {file.name}")

    cmd = [str(config.compiler_bin), "run", str(file)]
    if trace:
        cmd.extend(["--trace", trace])
    if emit_metrics:
        cmd.extend(["--emit-metrics", str(emit_metrics)])
    if args:
        cmd.extend(args)
    result = subprocess.run(cmd, cwd=config.apxm_dir, env=env)
    raise typer.Exit(result.returncode)


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
        print_info("  Build with: apxm build")

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
    trace: bool = typer.Option(False, "--trace", help="Build with tracing enabled (default)"),
    no_trace: bool = typer.Option(False, "--no-trace", help="Build with tracing disabled (zero overhead)"),
):
    """Build the APXM compiler.

    Tracing control:
      --trace     Build with runtime tracing enabled (default)
      --no-trace  Build with tracing compiled out (zero overhead for benchmarks)
    """
    config = get_config()
    conda_prefix = ensure_conda_env()
    env = setup_mlir_environment(conda_prefix, config.target_dir)

    # Handle conflicting options
    if trace and no_trace:
        print_error("Cannot specify both --trace and --no-trace")
        raise typer.Exit(1)

    # Build feature set (always include metrics for proper observability)
    features = ["driver", "metrics"]
    if no_trace:
        features.append("no-trace")
        print_header("Building APXM Compiler (no-trace)")
        print_info("Tracing disabled - zero overhead build")
    else:
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

    cmd = ["cargo", "build", "-p", "apxm-cli", "--features", ",".join(features)]
    if release:
        cmd.append("--release")

    result = subprocess.run(cmd, cwd=config.apxm_dir, env=env)

    if result.returncode == 0:
        print_success("Compiler built successfully!")
        print_info(f"Binary: {config.compiler_bin}")
        if no_trace:
            print_warning("Note: --trace flag at runtime will have no effect")
    else:
        print_error("Build failed!")
        raise typer.Exit(1)


# Workload Commands
@workloads_app.command("list")
def workloads_list():
    """List available benchmark workloads."""
    config = get_config()

    # Import consolidated runner for workload registry
    sys.path.insert(0, str(config.workloads_dir))
    try:
        from apxm_runner import list_workloads
        workloads = list_workloads()
    except ImportError:
        # Fall back to file-based discovery
        workloads = None
    finally:
        if str(config.workloads_dir) in sys.path:
            sys.path.remove(str(config.workloads_dir))

    print_header("Available Workloads")

    if workloads:
        # Use consolidated registry
        for w in workloads:
            console.print(f"  [{Colors.PASS}]{w['number']:2d}[/{Colors.PASS}] {w['directory']:30} {w['description']}")
            console.print(f"       [dim]Type: {w['type']}[/dim]")
        console.print()
        console.print(f"  Total: {len(workloads)} workloads")
    else:
        # Fall back to file-based discovery
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
    name: str = typer.Argument(..., help="Workload name or number (e.g., 10_multi_agent or 10)"),
    iterations: int = typer.Option(10, "-n", "--iterations", help="Benchmark iterations"),
    warmup: int = typer.Option(3, "-w", "--warmup", help="Warmup iterations"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON format"),
    apxm_only: bool = typer.Option(False, "--apxm-only", help="Run only A-PXM benchmark"),
    langgraph_only: bool = typer.Option(False, "--langgraph-only", help="Run only LangGraph benchmark"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Save results to file"),
):
    """Run a specific workload benchmark using consolidated runner."""
    import json as json_module

    config = get_config()
    conda_prefix = ensure_conda_env()
    env = setup_mlir_environment(conda_prefix, config.target_dir)

    # Set environment for subprocess
    os.environ.update(env)

    # Import consolidated runner
    sys.path.insert(0, str(config.workloads_dir))
    try:
        from apxm_runner import run_workload_benchmark, get_workload
    except ImportError as e:
        print_error(f"Could not import apxm_runner: {e}")
        raise typer.Exit(1)
    finally:
        if str(config.workloads_dir) in sys.path:
            sys.path.remove(str(config.workloads_dir))

    # Resolve workload
    workload = get_workload(name)
    if workload is None:
        # Try as integer
        try:
            workload = get_workload(int(name))
        except ValueError:
            pass

    if workload is None:
        print_error(f"Workload not found: {name}")
        print_info("Run 'workloads list' to see available workloads")
        raise typer.Exit(1)

    if not json_output:
        print_header(f"Running Workload: {workload.name}")
        print_info(f"Directory: {workload.directory}")
        print_info(f"Type: {workload.workload_type.value}")
        print_info(f"Iterations: {iterations}, Warmup: {warmup}")
        console.print()

    # Run benchmark
    run_lg = not apxm_only
    run_apxm = not langgraph_only

    results = run_workload_benchmark(
        workload.name,
        iterations=iterations,
        warmup=warmup,
        run_langgraph=run_lg,
        run_apxm=run_apxm,
    )

    # Output results
    if output:
        with open(output, "w") as f:
            json_module.dump(results, f, indent=2)
        if not json_output:
            print_success(f"Results saved to: {output}")

    if json_output:
        print(json_module.dumps(results, indent=2))
    elif not output:
        # Print summary
        lg_result = results.get("results", {}).get("langgraph", {})
        apxm_result = results.get("results", {}).get("apxm", {})

        console.print()
        print_header("Results")

        if lg_result:
            if "error" in lg_result:
                console.print(f"  LangGraph: [red]Error - {lg_result['error'][:60]}...[/red]")
            elif "mean_ms" in lg_result:
                console.print(f"  LangGraph: {lg_result['mean_ms']:.2f} ms")
            elif "note" in lg_result:
                console.print(f"  LangGraph: {lg_result['note']}")

        if apxm_result:
            if "error" in apxm_result:
                console.print(f"  A-PXM: [red]Error - {apxm_result['error'][:60]}...[/red]")
            elif "mean_ms" in apxm_result:
                console.print(f"  A-PXM: {apxm_result['mean_ms']:.2f} ms")
            elif apxm_result.get("success"):
                console.print(f"  A-PXM: Success")
            elif "note" in apxm_result:
                console.print(f"  A-PXM: {apxm_result['note']}")


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
