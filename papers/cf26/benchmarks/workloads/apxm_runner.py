#!/usr/bin/env python3
"""
A-PXM Benchmark Runner Utility

This module provides the proper way to run AIS workflows through the
full A-PXM pipeline (compiler → runtime) for benchmarking.

ALL benchmarks should use this module instead of simulating in Python.

Usage:
    from apxm_runner import run_ais_workflow, APXMConfig

    config = APXMConfig(opt_level=1)  # O1 for FuseReasoning, O0 without
    result = run_ais_workflow("workflow.ais", config)
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class APXMConfig:
    """Configuration for A-PXM execution."""
    opt_level: int = 1  # 0 = no FuseReasoning, 1+ = with FuseReasoning
    timeout_seconds: float = 120.0
    conda_prefix: Optional[str] = None

    def __post_init__(self):
        # Auto-detect CONDA_PREFIX if not provided
        if self.conda_prefix is None:
            self.conda_prefix = os.environ.get("CONDA_PREFIX")


@dataclass
class APXMResult:
    """Result from running an AIS workflow."""
    success: bool
    execution_time_ms: float
    total_nodes: int = 0
    compile_time_ms: float = 0.0
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    raw_output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "execution_time_ms": self.execution_time_ms,
            "total_nodes": self.total_nodes,
            "compile_time_ms": self.compile_time_ms,
            "error": self.error,
        }


def find_apxm_cli() -> Optional[Path]:
    """Find the apxm CLI binary."""
    # Check for cargo-built binary
    apxm_root = Path(__file__).parent.parent.parent.parent.parent
    release_bin = apxm_root / "target" / "release" / "apxm"
    debug_bin = apxm_root / "target" / "debug" / "apxm"

    if release_bin.exists():
        return release_bin
    if debug_bin.exists():
        return debug_bin

    # Check PATH
    try:
        result = subprocess.run(
            ["which", "apxm"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except Exception:
        pass

    return None


def build_apxm_cli() -> Optional[Path]:
    """Build the apxm CLI if not found."""
    apxm_root = Path(__file__).parent.parent.parent.parent.parent
    conda_prefix = os.environ.get("CONDA_PREFIX")

    if not conda_prefix:
        return None

    try:
        env = os.environ.copy()
        env["CONDA_PREFIX"] = conda_prefix

        result = subprocess.run(
            ["cargo", "build", "-p", "apxm-cli", "--features", "driver", "--release"],
            cwd=apxm_root,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )

        if result.returncode == 0:
            return apxm_root / "target" / "release" / "apxm"
    except Exception:
        pass

    return None


def run_ais_workflow(
    workflow_path: Path,
    config: Optional[APXMConfig] = None,
) -> APXMResult:
    """
    Run an AIS workflow through the full A-PXM pipeline.

    This is the CORRECT way to benchmark A-PXM - it goes through:
    1. DSL parsing
    2. MLIR generation
    3. Optimization passes (including FuseReasoning if opt_level > 0)
    4. Artifact generation
    5. Runtime execution with real LLM calls

    Args:
        workflow_path: Path to the .ais file
        config: Execution configuration

    Returns:
        APXMResult with timing and execution info
    """
    if config is None:
        config = APXMConfig()

    workflow_path = Path(workflow_path)
    if not workflow_path.exists():
        return APXMResult(
            success=False,
            execution_time_ms=0,
            error=f"Workflow file not found: {workflow_path}",
        )

    # Find or build CLI
    cli_path = find_apxm_cli()
    if cli_path is None:
        cli_path = build_apxm_cli()

    if cli_path is None:
        return APXMResult(
            success=False,
            execution_time_ms=0,
            error="Could not find or build apxm CLI. Set CONDA_PREFIX and run: cargo build -p apxm-cli --features driver --release",
        )

    # Build command
    cmd = [
        str(cli_path),
        "run",
        str(workflow_path),
        f"-O{config.opt_level}",
    ]

    # Set up environment
    env = os.environ.copy()
    if config.conda_prefix:
        env["CONDA_PREFIX"] = config.conda_prefix

    # Run with timing
    start_time = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            env=env,
        )
        execution_time_ms = (time.perf_counter() - start_time) * 1000

        # Parse output
        stdout = result.stdout
        stderr = result.stderr

        if result.returncode != 0:
            return APXMResult(
                success=False,
                execution_time_ms=execution_time_ms,
                stdout=stdout,
                stderr=stderr,
                error=f"CLI returned non-zero: {result.returncode}\n{stderr}",
                raw_output=stdout + stderr,
            )

        # Extract node count from output
        # Output format: "Executed N nodes in M ms"
        total_nodes = 0
        for line in stdout.split("\n"):
            if "nodes in" in line.lower():
                parts = line.split()
                for i, p in enumerate(parts):
                    if p.lower() == "executed" and i + 1 < len(parts):
                        try:
                            total_nodes = int(parts[i + 1])
                        except ValueError:
                            pass

        return APXMResult(
            success=True,
            execution_time_ms=execution_time_ms,
            total_nodes=total_nodes,
            stdout=stdout,
            stderr=stderr,
            raw_output=stdout,
        )

    except subprocess.TimeoutExpired:
        return APXMResult(
            success=False,
            execution_time_ms=config.timeout_seconds * 1000,
            error=f"Execution timed out after {config.timeout_seconds}s",
        )
    except Exception as e:
        return APXMResult(
            success=False,
            execution_time_ms=0,
            error=str(e),
        )


def run_benchmark(
    workflow_path: Path,
    config: Optional[APXMConfig] = None,
    iterations: int = 3,
    warmup: int = 1,
) -> Dict[str, Any]:
    """
    Run a benchmark with multiple iterations.

    Args:
        workflow_path: Path to the .ais file
        config: Execution configuration
        iterations: Number of benchmark iterations
        warmup: Number of warmup iterations (not counted)

    Returns:
        Dictionary with mean, std, min, max, samples
    """
    if config is None:
        config = APXMConfig()

    workflow_path = Path(workflow_path)

    # Warmup
    for _ in range(warmup):
        run_ais_workflow(workflow_path, config)

    # Benchmark
    samples = []
    errors = []

    for i in range(iterations):
        result = run_ais_workflow(workflow_path, config)
        if result.success:
            samples.append(result.execution_time_ms)
        else:
            errors.append(result.error)

    if not samples:
        return {
            "success": False,
            "error": errors[0] if errors else "No successful runs",
            "samples": [],
        }

    import statistics

    return {
        "success": True,
        "mean_ms": statistics.mean(samples),
        "std_ms": statistics.stdev(samples) if len(samples) > 1 else 0,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "p50_ms": statistics.median(samples),
        "samples": samples,
        "iterations": iterations,
        "opt_level": config.opt_level,
    }


def compare_optimization_levels(
    workflow_path: Path,
    iterations: int = 3,
) -> Dict[str, Any]:
    """
    Compare O0 (no FuseReasoning) vs O1 (with FuseReasoning).

    This is the key benchmark for FuseReasoning evaluation.
    """
    workflow_path = Path(workflow_path)

    print(f"Benchmarking: {workflow_path.name}")
    print(f"Iterations: {iterations}")
    print()

    # Run with O0 (no optimization)
    print("Running with O0 (no FuseReasoning)...")
    o0_config = APXMConfig(opt_level=0)
    o0_results = run_benchmark(workflow_path, o0_config, iterations)

    # Run with O1 (with FuseReasoning)
    print("Running with O1 (FuseReasoning enabled)...")
    o1_config = APXMConfig(opt_level=1)
    o1_results = run_benchmark(workflow_path, o1_config, iterations)

    # Calculate speedup
    speedup = None
    if o0_results.get("success") and o1_results.get("success"):
        o0_mean = o0_results["mean_ms"]
        o1_mean = o1_results["mean_ms"]
        if o1_mean > 0:
            speedup = o0_mean / o1_mean

    return {
        "workflow": str(workflow_path),
        "o0_unfused": o0_results,
        "o1_fused": o1_results,
        "speedup": speedup,
    }


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python apxm_runner.py <workflow.ais> [iterations]")
        sys.exit(1)

    workflow = Path(sys.argv[1])
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    results = compare_optimization_levels(workflow, iterations)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    if results["o0_unfused"].get("success"):
        print(f"O0 (Unfused): {results['o0_unfused']['mean_ms']:.0f} ms")
    else:
        print(f"O0 (Unfused): ERROR - {results['o0_unfused'].get('error')}")

    if results["o1_fused"].get("success"):
        print(f"O1 (Fused):   {results['o1_fused']['mean_ms']:.0f} ms")
    else:
        print(f"O1 (Fused):   ERROR - {results['o1_fused'].get('error')}")

    if results["speedup"]:
        print(f"\nSpeedup: {results['speedup']:.2f}x")
