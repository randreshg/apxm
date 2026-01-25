#!/usr/bin/env python3
"""
A-PXM Benchmark Runner Utility

This module provides the proper way to run AIS workflows through the
full A-PXM pipeline (compiler → runtime) for benchmarking.

ALL benchmarks should use this module instead of simulating in Python.

Usage:
    from apxm_runner import run_ais_workflow, APXMConfig, run_workload_benchmark

    # Run single workflow
    config = APXMConfig(opt_level=1)  # O1 for FuseAskOps, O0 without
    result = run_ais_workflow("workflow.ais", config)

    # Run full workload benchmark (consolidated entry point)
    result = run_workload_benchmark("1_parallel_research")
"""

import json
import re as _re
import os
import subprocess
import sys
import tempfile
import time
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from enum import Enum

# Add tools directory to path for shared utilities
_tools_dir = Path(__file__).parent.parent.parent.parent.parent / "tools"
sys.path.insert(0, str(_tools_dir))

from apxm_env import ApxmConfig as _ApxmConfig, get_conda_prefix, setup_mlir_environment


_METRICS_MARKER = ".apxm_metrics_build"


def _metrics_marker_path(config: _ApxmConfig) -> Path:
    return config.target_dir / "release" / _METRICS_MARKER


def _require_metrics() -> bool:
    return os.environ.get("APXM_BENCH_REQUIRE_METRICS", "1") != "0"


def _emit_diagnostics() -> bool:
    return os.environ.get("APXM_BENCH_EMIT_DIAGNOSTICS", "1") != "0"


def _keep_diagnostics() -> bool:
    return os.environ.get("APXM_BENCH_KEEP_DIAGNOSTICS", "0") == "1"


def _get_benchmark_settings():
    """Get benchmark settings from config file and environment.
    
    Imports lazily to avoid circular dependency.
    """
    try:
        from llm_instrumentation import get_benchmark_settings
        return get_benchmark_settings()
    except ImportError:
        return None


def _get_timeout(default: float = 120.0) -> float:
    """Get timeout from config file or APXM_BENCH_TIMEOUT env var."""
    # First try config file
    settings = _get_benchmark_settings()
    if settings is not None:
        return settings.timeout_seconds
    
    # Fall back to env var only
    timeout_str = os.environ.get("APXM_BENCH_TIMEOUT")
    if timeout_str:
        try:
            return float(timeout_str)
        except ValueError:
            pass
    return default


def calculate_framework_overhead(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate framework-specific overhead excluding LLM time.

    This helps isolate framework costs from LLM inference time,
    enabling fair comparison for workloads where LLM time dominates.

    Args:
        result: Dictionary with timing metrics (time in seconds, llm_latency_ms, compile_ms)

    Returns:
        Dictionary with:
            - framework_overhead_ms: Total time minus LLM time
            - compile_ms: Compilation time (if available)
            - llm_percentage: Percentage of total time spent in LLM calls
            - is_llm_bound: True if LLM time > 90% of total time
    """
    llm_time_ms = result.get("llm_latency_ms", 0) or 0
    total_time_ms = result.get("time", 0) * 1000  # Convert seconds to ms
    compile_time_ms = result.get("compile_ms", 0) or 0

    framework_overhead_ms = max(0, total_time_ms - llm_time_ms)
    llm_percentage = (llm_time_ms / total_time_ms * 100) if total_time_ms > 0 else 0

    return {
        "framework_overhead_ms": framework_overhead_ms,
        "compile_ms": compile_time_ms,
        "llm_percentage": llm_percentage,
        "is_llm_bound": llm_percentage > 90,
    }


def _run_compile_diagnostics(workflow_path: Path, opt_level: int) -> Dict[str, Any]:
    cli_path = find_apxm_cli(require_metrics=False)
    if cli_path is None:
        cli_path = build_apxm_cli(with_metrics=False)

    if cli_path is None:
        return {
            "success": False,
            "error": "Could not find apxm CLI. Build with: python tools/apxm_cli.py compiler build",
        }

    diag_tmp = tempfile.NamedTemporaryFile(prefix="apxm_diag_", suffix=".json", delete=False)
    diag_path = Path(diag_tmp.name)
    diag_tmp.close()

    out_tmp = tempfile.NamedTemporaryFile(prefix="apxm_artifact_", suffix=".apxmobj", delete=False)
    out_path = Path(out_tmp.name)
    out_tmp.close()

    cmd = [
        str(cli_path),
        "compile",
        str(workflow_path),
        f"-O{opt_level}",
        "--output",
        str(out_path),
        "--emit-diagnostics",
        str(diag_path),
    ]

    shared_config = _ApxmConfig.detect()
    if shared_config.conda_prefix:
        env = setup_mlir_environment(Path(shared_config.conda_prefix), shared_config.target_dir)
    else:
        env = os.environ.copy()

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)

    diagnostics: Optional[Dict[str, Any]] = None
    if diag_path.exists():
        try:
            diagnostics = json.loads(diag_path.read_text())
        except Exception:
            diagnostics = None

    if not _keep_diagnostics():
        for path in (diag_path, out_path):
            try:
                path.unlink()
            except OSError:
                pass

    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "diagnostics": diagnostics,
        "diagnostics_path": str(diag_path) if _keep_diagnostics() else None,
    }


@dataclass
class APXMConfig:
    """Configuration for A-PXM execution."""
    opt_level: int = 1  # 0 = no FuseAskOps, 1+ = with FuseAskOps
    timeout_seconds: float = 120.0
    conda_prefix: Optional[str] = None

    def __post_init__(self):
        # Override timeout from environment variable if set
        env_timeout = _get_timeout(self.timeout_seconds)
        if env_timeout != self.timeout_seconds:
            self.timeout_seconds = env_timeout
        
        # Auto-detect CONDA_PREFIX if not provided using shared utility
        if self.conda_prefix is None:
            prefix = get_conda_prefix()
            # Don't fall back to os.environ - it may be the base conda, not apxm
            self.conda_prefix = str(prefix) if prefix else None


@dataclass
class APXMResult:
    """Result from running an AIS workflow."""
    success: bool
    execution_time_ms: float
    total_nodes: int = 0
    compile_time_ms: float = 0.0
    runtime_ms: Optional[float] = None
    artifact_time_ms: Optional[float] = None
    validation_time_ms: Optional[float] = None
    llm_total_latency_ms: Optional[float] = None
    llm_requests: Optional[int] = None
    llm_total_input_tokens: Optional[int] = None
    llm_total_output_tokens: Optional[int] = None
    llm_avg_latency_ms: Optional[float] = None
    llm_p50_latency_ms: Optional[float] = None
    llm_p99_latency_ms: Optional[float] = None
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    raw_output: str = ""
    metrics: Optional[Dict[str, Any]] = None
    metrics_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "execution_time_ms": self.execution_time_ms,
            "total_nodes": self.total_nodes,
            "compile_time_ms": self.compile_time_ms,
            "runtime_ms": self.runtime_ms,
            "artifact_time_ms": self.artifact_time_ms,
            "validation_time_ms": self.validation_time_ms,
            "llm_total_latency_ms": self.llm_total_latency_ms,
            "llm_requests": self.llm_requests,
            "llm_total_input_tokens": self.llm_total_input_tokens,
            "llm_total_output_tokens": self.llm_total_output_tokens,
            "llm_avg_latency_ms": self.llm_avg_latency_ms,
            "llm_p50_latency_ms": self.llm_p50_latency_ms,
            "llm_p99_latency_ms": self.llm_p99_latency_ms,
            "error": self.error,
        }


def find_apxm_cli(require_metrics: bool = False) -> Optional[Path]:
    """Find the apxm CLI binary using shared configuration."""
    config = _ApxmConfig.detect()
    if config.compiler_bin.exists():
        if not require_metrics or _metrics_marker_path(config).exists():
            return config.compiler_bin

    # Fallback: check debug build
    debug_bin = config.target_dir / "debug" / "apxm"
    if debug_bin.exists() and not require_metrics:
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


def build_apxm_cli(with_metrics: bool = False) -> Optional[Path]:
    """Build the apxm CLI if not found."""
    config = _ApxmConfig.detect()
    conda_prefix = config.conda_prefix

    if not conda_prefix:
        return None

    try:
        env = setup_mlir_environment(conda_prefix, config.target_dir)

        features = "metrics" if with_metrics else "driver"
        result = subprocess.run(
            ["cargo", "build", "-p", "apxm-cli", "--features", features, "--release"],
            cwd=config.apxm_dir,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )

        if result.returncode == 0:
            if with_metrics:
                _metrics_marker_path(config).touch()
            return config.compiler_bin
    except Exception:
        pass

    return None


def run_ais_workflow(
    workflow_path: Path,
    config: Optional[APXMConfig] = None,
    capture_metrics: bool = True,
    args: Optional[List[str]] = None,
) -> APXMResult:
    """
    Run an AIS workflow through the full A-PXM pipeline.

    This is the CORRECT way to benchmark A-PXM - it goes through:
    1. DSL parsing
    2. MLIR generation
    3. Optimization passes (including FuseAskOps if opt_level > 0)
    4. Artifact generation
    5. Runtime execution with real LLM calls

    Args:
        workflow_path: Path to the .ais file
        config: Execution configuration
        args: Arguments to pass to the entry flow

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
    require_metrics = capture_metrics and _require_metrics()
    cli_path = find_apxm_cli(require_metrics=require_metrics)
    if cli_path is None:
        cli_path = build_apxm_cli(with_metrics=require_metrics)
    if cli_path is None and require_metrics:
        cli_path = build_apxm_cli(with_metrics=False)

    if cli_path is None:
        return APXMResult(
            success=False,
            execution_time_ms=0,
            error="Could not find apxm CLI. Build with: python tools/apxm_cli.py compiler build",
        )

    # Build command - use 'execute' for .ais source files
    # CLI expects: apxm execute [OPTIONS] FILE [ARGS]...
    # Options MUST come before the file path
    cmd = [
        str(cli_path),
        "execute",
        f"-O{config.opt_level}",
    ]
    metrics_path: Optional[Path] = None
    if capture_metrics:
        tmp = tempfile.NamedTemporaryFile(prefix="apxm_metrics_", suffix=".json", delete=False)
        metrics_path = Path(tmp.name)
        tmp.close()
        cmd.extend(["--emit-metrics", str(metrics_path)])
    # File path comes after options
    cmd.append(str(workflow_path))
    # Entry flow arguments come after file path
    if args:
        cmd.extend(args)

    # Set up environment using shared utility
    shared_config = _ApxmConfig.detect()
    if config.conda_prefix:
        env = setup_mlir_environment(Path(config.conda_prefix), shared_config.target_dir)
    elif shared_config.conda_prefix:
        # Use shared config's conda_prefix as fallback
        env = setup_mlir_environment(shared_config.conda_prefix, shared_config.target_dir)
    else:
        env = os.environ.copy()

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

        if os.environ.get("APXM_BENCH_VERBOSE") == "1":
            if stdout:
                print(stdout, end="")
            if stderr:
                print(stderr, end="", file=sys.stderr)

        metrics: Optional[Dict[str, Any]] = None
        runtime_ms: Optional[float] = None
        compile_ms: Optional[float] = None
        artifact_ms: Optional[float] = None
        validation_ms: Optional[float] = None
        llm_total_latency_ms: Optional[float] = None
        llm_requests: Optional[int] = None
        llm_total_input_tokens: Optional[int] = None
        llm_total_output_tokens: Optional[int] = None
        llm_avg_latency_ms: Optional[float] = None
        llm_p50_latency_ms: Optional[float] = None
        llm_p99_latency_ms: Optional[float] = None

        if metrics_path and metrics_path.exists():
            try:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)

                link_phases = metrics.get("link_phases", {})
                compile_ms = link_phases.get("compile_ms")
                artifact_ms = link_phases.get("artifact_ms")
                validation_ms = link_phases.get("validation_ms")
                runtime_ms = link_phases.get("runtime_ms")
                if runtime_ms is None:
                    runtime_ms = metrics.get("execution", {}).get("duration_ms")

                llm_metrics = metrics.get("llm")
                if llm_metrics:
                    def _safe_int(value: Any) -> Optional[int]:
                        if value is None:
                            return None
                        try:
                            return int(value)
                        except (TypeError, ValueError):
                            return None

                    def _safe_float(value: Any) -> Optional[float]:
                        if value is None:
                            return None
                        try:
                            return float(value)
                        except (TypeError, ValueError):
                            return None

                    llm_requests = _safe_int(llm_metrics.get("total_requests"))
                    llm_avg_latency_ms = _safe_float(llm_metrics.get("avg_latency_ms"))
                    llm_p50_latency_ms = _safe_float(llm_metrics.get("p50_latency_ms"))
                    llm_p99_latency_ms = _safe_float(llm_metrics.get("p99_latency_ms"))
                    llm_total_input_tokens = _safe_int(llm_metrics.get("total_input_tokens"))
                    llm_total_output_tokens = _safe_int(llm_metrics.get("total_output_tokens"))
                    if llm_requests and llm_avg_latency_ms is not None:
                        llm_total_latency_ms = llm_avg_latency_ms * llm_requests
            except Exception:
                metrics = None

        keep_metrics = os.environ.get("APXM_BENCH_KEEP_METRICS") == "1"
        if metrics_path and not keep_metrics:
            try:
                metrics_path.unlink()
            except OSError:
                pass

        if result.returncode != 0:
            return APXMResult(
                success=False,
                execution_time_ms=execution_time_ms,
                compile_time_ms=compile_ms or 0.0,
                runtime_ms=runtime_ms,
                artifact_time_ms=artifact_ms,
                validation_time_ms=validation_ms,
                llm_total_latency_ms=llm_total_latency_ms,
                llm_requests=llm_requests,
                llm_total_input_tokens=llm_total_input_tokens,
                llm_total_output_tokens=llm_total_output_tokens,
                llm_avg_latency_ms=llm_avg_latency_ms,
                llm_p50_latency_ms=llm_p50_latency_ms,
                llm_p99_latency_ms=llm_p99_latency_ms,
                stdout=stdout,
                stderr=stderr,
                error=f"CLI returned non-zero: {result.returncode}\n{stderr}",
                raw_output=stdout + stderr,
                metrics=metrics,
                metrics_path=str(metrics_path) if keep_metrics and metrics_path else None,
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
            compile_time_ms=compile_ms or 0.0,
            runtime_ms=runtime_ms,
            artifact_time_ms=artifact_ms,
            validation_time_ms=validation_ms,
            llm_total_latency_ms=llm_total_latency_ms,
            llm_requests=llm_requests,
            llm_total_input_tokens=llm_total_input_tokens,
            llm_total_output_tokens=llm_total_output_tokens,
            llm_avg_latency_ms=llm_avg_latency_ms,
            llm_p50_latency_ms=llm_p50_latency_ms,
            llm_p99_latency_ms=llm_p99_latency_ms,
            stdout=stdout,
            stderr=stderr,
            raw_output=stdout,
            metrics=metrics,
            metrics_path=str(metrics_path) if keep_metrics and metrics_path else None,
        )

    except subprocess.TimeoutExpired:
        if metrics_path and metrics_path.exists() and os.environ.get("APXM_BENCH_KEEP_METRICS") != "1":
            try:
                metrics_path.unlink()
            except OSError:
                pass
        return APXMResult(
            success=False,
            execution_time_ms=config.timeout_seconds * 1000,
            error=f"Execution timed out after {config.timeout_seconds}s",
        )
    except Exception as e:
        if metrics_path and metrics_path.exists() and os.environ.get("APXM_BENCH_KEEP_METRICS") != "1":
            try:
                metrics_path.unlink()
            except OSError:
                pass
        return APXMResult(
            success=False,
            execution_time_ms=0,
            error=str(e),
        )


def run_benchmark(
    workflow_path: Path,
    config: Optional[APXMConfig] = None,
    iterations: int = 10,
    warmup: int = 3,
    args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run a benchmark with multiple iterations.

    Args:
        workflow_path: Path to the .ais file
        config: Execution configuration
        iterations: Number of benchmark iterations
        warmup: Number of warmup iterations (not counted)
        args: Arguments to pass to the entry flow

    Returns:
        Dictionary with mean, std, min, max, samples
    """
    if config is None:
        config = APXMConfig()

    workflow_path = Path(workflow_path)

    compiler_diagnostics = None
    if _emit_diagnostics():
        compiler_diagnostics = _run_compile_diagnostics(workflow_path, config.opt_level)

    # Warmup
    for _ in range(warmup):
        run_ais_workflow(workflow_path, config, capture_metrics=False, args=args)

    # Benchmark
    samples: List[float] = []
    sample_details: List[Dict[str, Any]] = []
    metrics_samples: List[Dict[str, float]] = []
    runtime_ms_values: List[float] = []
    compile_ms_values: List[float] = []
    artifact_ms_values: List[float] = []
    validation_ms_values: List[float] = []
    llm_total_ms_values: List[float] = []
    llm_requests_values: List[float] = []
    runtime_non_llm_ms_values: List[float] = []
    llm_input_tokens_values: List[float] = []
    llm_output_tokens_values: List[float] = []
    llm_avg_latency_values: List[float] = []
    llm_p50_latency_values: List[float] = []
    llm_p99_latency_values: List[float] = []
    errors = []

    for i in range(iterations):
        iteration_ts = datetime.now(timezone.utc).isoformat()
        result = run_ais_workflow(workflow_path, config, args=args)

        sample_entry: Dict[str, Any] = {
            "iteration": i,
            "timestamp": iteration_ts,
            "success": result.success,
            "wall_time_ms": result.execution_time_ms,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if result.error:
            sample_entry["error"] = result.error
        if result.metrics is not None:
            sample_entry["runtime_metrics"] = result.metrics
        sample_entry["phases"] = {
            "compile_ms": result.compile_time_ms,
            "artifact_ms": result.artifact_time_ms,
            "validation_ms": result.validation_time_ms,
            "runtime_ms": result.runtime_ms,
        }
        sample_entry["llm"] = {
            "total_latency_ms": result.llm_total_latency_ms,
            "requests": result.llm_requests,
            "input_tokens": result.llm_total_input_tokens,
            "output_tokens": result.llm_total_output_tokens,
            "avg_latency_ms": result.llm_avg_latency_ms,
            "p50_latency_ms": result.llm_p50_latency_ms,
            "p99_latency_ms": result.llm_p99_latency_ms,
        }
        sample_details.append(sample_entry)

        if result.success:
            samples.append(result.execution_time_ms)
            if result.metrics is not None:
                runtime_ms = result.runtime_ms
                llm_total_ms = result.llm_total_latency_ms
                llm_requests = result.llm_requests
                if runtime_ms is not None:
                    runtime_ms_values.append(runtime_ms)
                if result.compile_time_ms:
                    compile_ms_values.append(result.compile_time_ms)
                if result.artifact_time_ms is not None:
                    artifact_ms_values.append(result.artifact_time_ms)
                if result.validation_time_ms is not None:
                    validation_ms_values.append(result.validation_time_ms)
                if llm_total_ms is not None:
                    llm_total_ms_values.append(llm_total_ms)
                if llm_requests is not None:
                    llm_requests_values.append(float(llm_requests))
                if result.llm_total_input_tokens is not None:
                    llm_input_tokens_values.append(float(result.llm_total_input_tokens))
                if result.llm_total_output_tokens is not None:
                    llm_output_tokens_values.append(float(result.llm_total_output_tokens))
                if result.llm_avg_latency_ms is not None:
                    llm_avg_latency_values.append(result.llm_avg_latency_ms)
                if result.llm_p50_latency_ms is not None:
                    llm_p50_latency_values.append(result.llm_p50_latency_ms)
                if result.llm_p99_latency_ms is not None:
                    llm_p99_latency_values.append(result.llm_p99_latency_ms)
                if runtime_ms is not None and llm_total_ms is not None:
                    llm_wall_ms = min(llm_total_ms, runtime_ms)
                    runtime_non_llm_ms_values.append(max(runtime_ms - llm_wall_ms, 0.0))

                metrics_samples.append({
                    "wall_time_ms": result.execution_time_ms,
                    "runtime_ms": runtime_ms or 0.0,
                    "compile_ms": result.compile_time_ms,
                    "artifact_ms": result.artifact_time_ms or 0.0,
                    "validation_ms": result.validation_time_ms or 0.0,
                    "llm_total_ms": llm_total_ms or 0.0,
                    "llm_requests": float(llm_requests or 0),
                    "llm_input_tokens": float(result.llm_total_input_tokens or 0),
                    "llm_output_tokens": float(result.llm_total_output_tokens or 0),
                    "llm_avg_latency_ms": float(result.llm_avg_latency_ms or 0.0),
                    "llm_p50_latency_ms": float(result.llm_p50_latency_ms or 0.0),
                    "llm_p99_latency_ms": float(result.llm_p99_latency_ms or 0.0),
                })
        else:
            errors.append(result.error)

    if not samples:
        return {
            "success": False,
            "error": errors[0] if errors else "No successful runs",
            "samples": [],
            "sample_details": sample_details,
            "compiler": compiler_diagnostics,
            "metrics_samples": metrics_samples,
        }

    import statistics

    def _percentile(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        if pct <= 0:
            return ordered[0]
        if pct >= 100:
            return ordered[-1]
        idx = (len(ordered) - 1) * (pct / 100.0)
        lo = int(idx)
        hi = min(lo + 1, len(ordered) - 1)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)

    def _stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {}
        return {
            "mean_ms": statistics.mean(values),
            "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min_ms": min(values),
            "max_ms": max(values),
            "p50_ms": statistics.median(values),
            "p95_ms": _percentile(values, 95),
        }

    return {
        "success": True,
        "mean_ms": statistics.mean(samples),
        "std_ms": statistics.stdev(samples) if len(samples) > 1 else 0,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": _percentile(samples, 95),
        "samples": samples,
        "sample_details": sample_details,
        "iterations": iterations,
        "opt_level": config.opt_level,
        "compiler": compiler_diagnostics,
        "metrics": {
            "runtime_ms": _stats(runtime_ms_values),
            "compile_ms": _stats(compile_ms_values),
            "artifact_ms": _stats(artifact_ms_values),
            "validation_ms": _stats(validation_ms_values),
            "llm_total_ms": _stats(llm_total_ms_values),
            "llm_requests": _stats(llm_requests_values),
            "runtime_non_llm_ms": _stats(runtime_non_llm_ms_values),
            "llm_input_tokens": _stats(llm_input_tokens_values),
            "llm_output_tokens": _stats(llm_output_tokens_values),
            "llm_avg_latency_ms": _stats(llm_avg_latency_values),
            "llm_p50_latency_ms": _stats(llm_p50_latency_values),
            "llm_p99_latency_ms": _stats(llm_p99_latency_values),
        },
        "metrics_samples": metrics_samples,
    }


def compare_optimization_levels(
    workflow_path: Path,
    iterations: int = 10,
) -> Dict[str, Any]:
    """
    Compare O0 (no FuseAskOps) vs O1 (with FuseAskOps).

    This is the key benchmark for FuseAskOps evaluation.
    """
    workflow_path = Path(workflow_path)

    print(f"Benchmarking: {workflow_path.name}")
    print(f"Iterations: {iterations}")
    print()

    # Run with O0 (no optimization)
    print("Running with O0 (no FuseAskOps)...")
    o0_config = APXMConfig(opt_level=0)
    o0_results = run_benchmark(workflow_path, o0_config, iterations)

    # Run with O1 (with FuseAskOps)
    print("Running with O1 (FuseAskOps enabled)...")
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


# =============================================================================
# WORKLOAD CONFIGURATION REGISTRY
# =============================================================================

class WorkloadType(Enum):
    """Types of workload benchmarks with different execution patterns."""
    STANDARD = "standard"           # Normal A-PXM vs LangGraph comparison
    ERROR_DETECTION = "error"       # Compile-time vs runtime error detection
    SCALABILITY = "scalability"     # N-way parallelism (N=2,4,8)
    COMPILE_ONLY = "compile"        # Compilation scaling (no LLM calls)
    FUSION_COMPARISON = "fusion"    # O0 vs O1 comparison
    LLM_PROBE = "probe"             # Single LLM call measurement
    TOKEN_ESTIMATION = "tokens"     # Token cost estimation
    DATASET = "dataset"             # Dataset-driven benchmarks (accuracy/F1 metrics)


@dataclass
class WorkloadConfig:
    """Configuration for a single workload benchmark."""
    name: str
    directory: str
    description: str
    workload_type: WorkloadType = WorkloadType.STANDARD
    workflow_file: str = "workflow.ais"
    opt_level: int = 1
    default_iterations: int = 10
    default_warmup: int = 3
    # For LangGraph initial state
    initial_state: Dict[str, Any] = field(default_factory=dict)
    # Entry flow arguments for A-PXM CLI
    entry_args: List[str] = field(default_factory=list)
    # For scalability workloads
    parallel_levels: List[int] = field(default_factory=list)
    # For compilation scaling
    op_counts: List[int] = field(default_factory=list)
    # Extra workflow files (for scalability, fusion quality, etc.)
    extra_workflows: Dict[str, str] = field(default_factory=dict)
    # For dataset workloads
    dataset_name: Optional[str] = None  # "hotpotqa", "parallelqa", "movie"
    max_samples: Optional[int] = None  # Limit dataset size for benchmarks


# Workload registry - current workloads
WORKLOADS: Dict[int, WorkloadConfig] = {
    1: WorkloadConfig(
        name="parallel_research",
        directory="1_parallel_research",
        description="Automatic parallelism from dataflow",
        initial_state={"topic": "quantum computing", "background": "", "advances": "", "impact": "", "combined": "", "report": ""},
        entry_args=["quantum computing"],
    ),
    2: WorkloadConfig(
        name="chain_fusion",
        directory="2_chain_fusion",
        description="FuseAskOps compiler optimization (5 calls -> 1)",
        workload_type=WorkloadType.FUSION_COMPARISON,
        initial_state={"step1": "", "step2": "", "step3": "", "step4": "", "summary": ""},
        # main() has no arguments
    ),
    3: WorkloadConfig(
        name="type_verification",
        directory="3_type_verification",
        description="Compile-time vs runtime error detection",
        workload_type=WorkloadType.ERROR_DETECTION,
        initial_state={"result": "", "output": ""},
        # main() has no arguments
    ),
    4: WorkloadConfig(
        name="scalability",
        directory="4_scalability",
        description="N-way parallelism efficiency (N=2,4,8)",
        workload_type=WorkloadType.SCALABILITY,
        parallel_levels=[2, 4, 8],
        extra_workflows={"2": "workflow_n2.ais", "4": "workflow_n4.ais", "8": "workflow_n8.ais"},
        # main() has no arguments
    ),
    5: WorkloadConfig(
        name="memory_augmented",
        directory="5_memory_augmented",
        description="3-tier memory system (STM/LTM/Episodic)",
        initial_state={
            "query": "What is machine learning?",
            "stm": {},
            "ltm": {"domain_knowledge": "Machine learning fundamentals"},
            "episodic": [],
            "cached": "",
            "answer": "",
        },
        entry_args=["What is machine learning?"],
    ),
    6: WorkloadConfig(
        name="tool_invocation",
        directory="6_tool_invocation",
        description="Native INV operations with capability system",
        initial_state={"query": "Search for recent AI news", "tool_result": "", "answer": ""},
        entry_args=["Search for recent AI news"],
    ),
    7: WorkloadConfig(
        name="reflection",
        directory="7_reflection",
        description="Built-in reflect operation for self-improvement",
        initial_state={"task": "Write a haiku about coding", "initial_answer": "", "reflection": "", "improved_answer": ""},
        entry_args=["Write a haiku about coding"],
    ),
    8: WorkloadConfig(
        name="planning",
        directory="8_planning",
        description="Native plan operation for task decomposition",
        initial_state={"goal": "Build a simple web app", "steps": "", "step1_result": "", "step2_result": "", "step3_result": "", "combined": "", "final_result": ""},
        entry_args=["Build a simple web app"],
    ),
    9: WorkloadConfig(
        name="conditional_routing",
        directory="9_conditional_routing",
        description="Dataflow-based parallel preparation and routing",
        initial_state={"input": "Explain quantum entanglement", "category": "", "output": ""},
        entry_args=["Explain quantum entanglement"],
    ),
    10: WorkloadConfig(
        name="multi_agent",
        directory="10_multi_agent",
        description="Native agent definitions with communicate operations",
        initial_state={"topic": "climate change", "research_result": "", "critique_prep": "", "critique_result": "", "final_report": ""},
        entry_args=["climate change"],
    ),
}

# Name lookup for convenience
WORKLOAD_BY_NAME: Dict[str, int] = {w.name: num for num, w in WORKLOADS.items()}
WORKLOAD_BY_DIR: Dict[str, int] = {w.directory: num for num, w in WORKLOADS.items()}


def get_workload(identifier: str | int) -> Optional[WorkloadConfig]:
    """Get workload config by number, name, or directory."""
    if isinstance(identifier, int):
        return WORKLOADS.get(identifier)
    # Try name first
    if identifier in WORKLOAD_BY_NAME:
        return WORKLOADS[WORKLOAD_BY_NAME[identifier]]
    # Try directory
    if identifier in WORKLOAD_BY_DIR:
        return WORKLOADS[WORKLOAD_BY_DIR[identifier]]
    # Try partial match on directory (e.g., "1_parallel" matches "1_parallel_research")
    for dir_name, num in WORKLOAD_BY_DIR.items():
        if identifier in dir_name or dir_name.startswith(identifier):
            return WORKLOADS[num]
    return None


def _get_workload_dir(workload: WorkloadConfig) -> Path:
    """Get the directory path for a workload."""
    return Path(__file__).parent / workload.directory


def _load_langgraph_module(workload: WorkloadConfig):
    """Dynamically load the LangGraph workflow module."""
    workload_dir = _get_workload_dir(workload)
    workflow_py = workload_dir / "workflow.py"

    if not workflow_py.exists():
        return None

    import importlib.util
    spec = importlib.util.spec_from_file_location("workflow", workflow_py)
    module = importlib.util.module_from_spec(spec)

    # Add workload dir to path for imports
    sys.path.insert(0, str(workload_dir))
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        if str(workload_dir) in sys.path:
            sys.path.remove(str(workload_dir))


# =============================================================================
# WORKLOAD TYPE HANDLERS
# =============================================================================

def _run_standard_langgraph(workload: WorkloadConfig, iterations: int, warmup: int) -> Dict[str, Any]:
    """Run standard LangGraph benchmark."""
    module = _load_langgraph_module(workload)
    if module is None:
        return {"error": "workflow.py not found"}

    if not hasattr(module, "graph"):
        return {"error": "No 'graph' found in workflow.py"}

    # Import langgraph_bench helper
    from langgraph_bench import run_graph

    has_ollama = getattr(module, "HAS_OLLAMA", False)
    result = run_graph(module.graph, workload.initial_state.copy(), iterations, warmup)
    result["has_ollama"] = has_ollama
    return result


def _run_standard_apxm(workload: WorkloadConfig, iterations: int, warmup: int) -> Dict[str, Any]:
    """Run standard A-PXM benchmark."""
    workflow_file = _get_workload_dir(workload) / workload.workflow_file
    if not workflow_file.exists():
        return {"error": f"Workflow file not found: {workflow_file}"}

    config = APXMConfig(opt_level=workload.opt_level)
    # Pass entry flow arguments if defined
    args = workload.entry_args if workload.entry_args else None
    return run_benchmark(workflow_file, config, iterations, warmup=warmup, args=args)


def _run_error_detection_langgraph(workload: WorkloadConfig, iterations: int, warmup: int) -> Dict[str, Any]:
    """Run LangGraph error detection test."""
    module = _load_langgraph_module(workload)
    if module is None:
        return {"error": "workflow.py not found"}

    from llm_instrumentation import consume_latencies_ms, reset_metrics, summarize_latencies

    has_ollama = getattr(module, "HAS_OLLAMA", False)

    reset_metrics()
    start = time.perf_counter()
    try:
        module.graph.invoke(workload.initial_state.copy())
        llm_summary = summarize_latencies(consume_latencies_ms())
        return {
            "error_caught": False,
            "error_type": None,
            "time_to_error_ms": 0,
            "llm_calls_before_error": 0,
            "llm": llm_summary,
            "has_ollama": has_ollama,
        }
    except KeyError as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        llm_summary = summarize_latencies(consume_latencies_ms())
        return {
            "error_caught": True,
            "error_type": "KeyError",
            "error_message": str(e),
            "time_to_error_ms": elapsed_ms,
            "llm_calls_before_error": 1,
            "cost_wasted": "$0.01+",
            "has_ollama": has_ollama,
            "llm": llm_summary,
        }


def _run_error_detection_apxm(workload: WorkloadConfig, iterations: int, warmup: int) -> Dict[str, Any]:
    """Run A-PXM compile-time error detection test."""
    cli_path = find_apxm_cli()
    workflow_file = _get_workload_dir(workload) / workload.workflow_file

    if cli_path is None:
        return {
            "error_caught": None,
            "note": "A-PXM CLI not available - build with: python tools/apxm_cli.py compiler build",
            "expected_behavior": {
                "error_type": "compile-time",
                "time_to_error_ms": "~50ms",
                "llm_calls_before_error": 0,
                "cost_wasted": "$0.00",
            },
        }

    config = _ApxmConfig.detect()
    env = None
    if config.conda_prefix:
        env = setup_mlir_environment(config.conda_prefix, config.target_dir)

    start = time.perf_counter()
    # CLI expects: apxm execute [OPTIONS] FILE [ARGS]...
    result = subprocess.run(
        [str(cli_path), "execute", "-O0", str(workflow_file)],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    if result.returncode != 0:
        return {
            "error_caught": True,
            "error_type": "compile-time",
            "error_message": result.stderr.strip()[:200],
            "time_to_error_ms": elapsed_ms,
            "llm_calls_before_error": 0,
            "cost_wasted": "$0.00",
        }
    else:
        return {
            "error_caught": False,
            "note": "Execution succeeded unexpectedly",
        }


def _run_scalability_langgraph(workload: WorkloadConfig, iterations: int, warmup: int) -> Dict[str, Any]:
    """Run LangGraph scalability benchmark at different N levels."""
    module = _load_langgraph_module(workload)
    if module is None:
        return {"error": "workflow.py not found"}

    if not hasattr(module, "build_parallel_graph"):
        return {"error": "No 'build_parallel_graph' found in workflow.py"}

    from langgraph_bench import run_graph

    has_ollama = getattr(module, "HAS_OLLAMA", False)
    series = []

    for n in workload.parallel_levels:
        graph = module.build_parallel_graph(n)
        initial_state = {"results": [], "final": ""}
        metrics = run_graph(graph, initial_state, iterations, warmup)

        series.append({
            "n": n,
            "T_1": n,
            "T_inf": 1,
            "theoretical_speedup": n,
            "has_ollama": has_ollama,
            **metrics,
        })

    return {"series": series, "parallel_levels": workload.parallel_levels}


def _run_scalability_apxm(workload: WorkloadConfig, iterations: int, warmup: int) -> Dict[str, Any]:
    """Run A-PXM scalability benchmark at different N levels."""
    workload_dir = _get_workload_dir(workload)
    series = []

    for n in workload.parallel_levels:
        workflow_key = str(n)
        if workflow_key not in workload.extra_workflows:
            series.append({"n": n, "error": f"No workflow file for N={n}"})
            continue

        workflow_file = workload_dir / workload.extra_workflows[workflow_key]
        if not workflow_file.exists():
            series.append({"n": n, "error": f"Workflow file not found: {workflow_file}"})
            continue

        config = APXMConfig(opt_level=1)
        result = run_benchmark(workflow_file, config, iterations, warmup=warmup)

        if result.get("success"):
            series.append({
                "n": n,
                "T_1": n,
                "T_inf": 1,
                "theoretical_speedup": n,
                "success": True,
                **{k: v for k, v in result.items() if k != "success"},
            })
        else:
            series.append({"n": n, "error": result.get("error", "Unknown error")})

    return {"series": series, "parallel_levels": workload.parallel_levels}


def _run_compile_only_apxm(workload: WorkloadConfig, iterations: int, warmup: int) -> Dict[str, Any]:
    """Run compilation scaling benchmark (no LLM calls)."""
    cli_path = find_apxm_cli()
    if cli_path is None:
        return {"error": "A-PXM CLI not available"}

    def generate_synthetic_ais(num_ops: int) -> str:
        lines = [f"// Synthetic AIS file with {num_ops} operations", "", "agent ScalingTest {", "    flow main {"]
        result_names = []
        for i in range(num_ops):
            result_name = f"result_{i}"
            result_names.append(result_name)
            lines.append(f'        ask "Task {i}: Process data element" -> {result_name}')
        lines.append("")
        lines.append(f"        merge [{', '.join(result_names)}] -> final")
        lines.extend(["    }", "}"])
        return "\n".join(lines)

    config = _ApxmConfig.detect()
    env = setup_mlir_environment(config.conda_prefix, config.target_dir) if config.conda_prefix else os.environ.copy()

    results = []
    for num_ops in workload.op_counts:
        ais_content = generate_synthetic_ais(num_ops)
        samples = []

        for i in range(warmup + iterations):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                ais_file = tmpdir / "test.ais"
                diag_file = tmpdir / "diagnostics.json"
                output_file = tmpdir / "output.apxmobj"
                ais_file.write_text(ais_content)

                cmd = [str(cli_path), "compile", str(ais_file), "-O1", "--emit-diagnostics", str(diag_file), "-o", str(output_file)]
                start = time.perf_counter()
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
                wall_time_ms = (time.perf_counter() - start) * 1000

                if i >= warmup:
                    sample = {"wall_time_ms": wall_time_ms, "success": result.returncode == 0}
                    if result.returncode == 0 and diag_file.exists():
                        sample["diagnostics"] = json.loads(diag_file.read_text())
                    samples.append(sample)

        successful = [s for s in samples if s["success"]]
        if successful:
            wall_times = [s["wall_time_ms"] for s in successful]
            results.append({
                "num_ops": num_ops,
                "mean_ms": statistics.mean(wall_times),
                "std_ms": statistics.stdev(wall_times) if len(wall_times) > 1 else 0,
                "samples": samples,
            })
        else:
            results.append({"num_ops": num_ops, "error": "No successful runs"})

    return {"op_counts": workload.op_counts, "results": results}


def _run_compile_only_langgraph(workload: WorkloadConfig, iterations: int, warmup: int) -> Dict[str, Any]:
    """LangGraph has no compilation phase."""
    return {"note": "LangGraph has no compilation phase"}


def _run_dataset_apxm(workload: WorkloadConfig, iterations: int, warmup: int) -> Dict[str, Any]:
    """
    Run A-PXM dataset benchmark.

    For dataset workloads, iterations is interpreted as max_samples (dataset size limit).
    Each example is run once through the workflow.
    """
    from dataset_eval import (
        load_movie_dataset,
        evaluate_dataset_results,
    )
    
    # Load dataset
    dataset_name = workload.dataset_name
    if dataset_name == "movie":
        dataset = load_movie_dataset(max_samples=workload.max_samples or iterations)
    else:
        return {"error": f"Unknown dataset: {dataset_name}"}
    
    if not dataset:
        return {"error": "Dataset is empty"}
    
    # Get workflow file
    workload_dir = _get_workload_dir(workload)
    workflow_file = workload_dir / workload.workflow_file
    if not workflow_file.exists():
        return {"error": f"Workflow file not found: {workflow_file}"}
    
    config = APXMConfig(opt_level=workload.opt_level)
    
    # Run each example
    results = {}
    latencies = []
    errors = []
    
    for example in dataset:
        example_id = str(example.get("id", len(results)))
        question = example.get("question", "")
        label = example.get("answer", "")
        
        if not question:
            continue
        
        # Run workflow with question as argument
        # For A-PXM, we pass the question as entry flow argument
        result = run_ais_workflow(
            workflow_file,
            config,
            capture_metrics=True,
            args=[question] if question else None,
        )
        
        if result.success:
            # Extract answer from output (last line or stdout)
            answer = result.raw_output.strip()
            # Try to extract from last non-empty line, filtering out log messages
            lines = [l.strip() for l in answer.split("\n") if l.strip()]
            # Filter out log lines (Wrote metrics, Executing workflow, etc.)
            lines = [l for l in lines if not l.startswith("Wrote ") and not l.startswith("┏") and not l.startswith("┃") and not l.startswith("┗")]

            if lines:
                # Look for "Answer: X" pattern from print statements
                for line in reversed(lines):
                    if line.startswith("Answer: "):
                        answer = line[len("Answer: "):].strip()
                        break
                else:
                    answer = lines[-1]


                answer = answer.strip().rstrip('.').strip()
                answer = _re.sub(r'^[-•*]\s*', '', answer)
                answer = answer.split('\n')[0].strip()

            # For fairness vs LangGraph (in-process invoke timing), prefer A-PXM runtime_ms
            # when available. execution_time_ms includes subprocess + compile overhead.
            time_s = (result.runtime_ms / 1000.0) if result.runtime_ms is not None else (result.execution_time_ms / 1000.0)
            example_result = {
                "question": question,
                "answer": answer,
                "label": label,
                "time": time_s,  # seconds
                # LLM metrics
                "llm_latency_ms": result.llm_total_latency_ms,
                "llm_requests": result.llm_requests,
                "input_tokens": result.llm_total_input_tokens,
                "output_tokens": result.llm_total_output_tokens,
                "compile_ms": result.compile_time_ms,
            }
            # Add framework overhead metrics
            overhead = calculate_framework_overhead(example_result)
            example_result.update(overhead)
            results[example_id] = example_result
            latencies.append(result.execution_time_ms)
        else:
            errors.append(f"Example {example_id}: {result.error}")
            results[example_id] = {
                "question": question,
                "answer": "",
                "label": label,
                "error": result.error,
            }
    
    # Evaluate metrics
    metrics = evaluate_dataset_results(results, max_samples=None)
    
    return {
        "success": len(errors) == 0 or len(results) > 0,
        "num_examples": len(dataset),
        "num_successful": len([r for r in results.values() if "error" not in r]),
        "num_errors": len(errors),
        "errors": errors[:10],  # Limit error list
        "metrics": metrics,
        "results": results,
    }


def _run_dataset_langgraph(workload: WorkloadConfig, iterations: int, warmup: int) -> Dict[str, Any]:
    """
    Run LangGraph dataset benchmark.

    For dataset workloads, iterations is interpreted as max_samples (dataset size limit).
    Each example is run once through the workflow.
    """
    from dataset_eval import (
        load_movie_dataset,
        evaluate_dataset_results,
    )
    
    # Load dataset
    dataset_name = workload.dataset_name
    if dataset_name == "movie":
        dataset = load_movie_dataset(max_samples=workload.max_samples or iterations)
    else:
        return {"error": f"Unknown dataset: {dataset_name}"}
    
    if not dataset:
        return {"error": "Dataset is empty"}
    
    # Load LangGraph module
    module = _load_langgraph_module(workload)
    if module is None:
        return {"error": "workflow.py not found"}
    
    if not hasattr(module, "graph"):
        return {"error": "No 'graph' found in workflow.py"}
    
    from langgraph_bench import run_graph
    from llm_instrumentation import reset_metrics, consume_metrics
    
    has_ollama = getattr(module, "HAS_OLLAMA", False)
    
    # Run each example
    results = {}
    latencies = []
    errors = []
    
    for example in dataset:
        example_id = str(example.get("id", len(results)))
        question = example.get("question", "")
        label = example.get("answer", "")
        
        if not question:
            continue
        
        # Prepare initial state with question
        # Most LangGraph workflows expect a "question" or "query" field
        initial_state = workload.initial_state.copy()
        if "question" in initial_state:
            initial_state["question"] = question
        elif "query" in initial_state:
            initial_state["query"] = question
        else:
            # Try to infer from workload config
            initial_state = {"question": question, **initial_state}
        
        # Run graph with LLM instrumentation
        try:
            reset_metrics()
            start_time = time.perf_counter()
            output = module.graph.invoke(initial_state)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            # Collect LLM metrics
            llm_calls = consume_metrics()
            llm_latency_ms = sum(c.get("latency_ms", 0) for c in llm_calls) if llm_calls else None
            input_tokens = sum(c.get("input_tokens", 0) or 0 for c in llm_calls) if llm_calls else 0
            output_tokens = sum(c.get("output_tokens", 0) or 0 for c in llm_calls) if llm_calls else 0
            
            # Extract answer from output
            # Try common field names
            answer = ""
            if isinstance(output, dict):
                answer = output.get("answer", output.get("output", output.get("result", "")))
                if not answer:
                    # Get last non-empty string value
                    for v in reversed(output.values()):
                        if isinstance(v, str) and v.strip():
                            answer = v.strip()
                            break
            elif isinstance(output, str):
                answer = output
            
            example_result = {
                "question": question,
                "answer": answer,
                "label": label,
                "time": elapsed_ms / 1000.0,  # Convert to seconds
                # LLM metrics
                "llm_latency_ms": llm_latency_ms,
                "llm_requests": len(llm_calls) if llm_calls else 0,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            # Add framework overhead metrics
            overhead = calculate_framework_overhead(example_result)
            example_result.update(overhead)
            results[example_id] = example_result
            latencies.append(elapsed_ms)
        except Exception as e:
            errors.append(f"Example {example_id}: {str(e)}")
            results[example_id] = {
                "question": question,
                "answer": "",
                "label": label,
                "error": str(e),
            }
    
    # Evaluate metrics
    metrics = evaluate_dataset_results(results, max_samples=None)
    
    return {
        "success": len(errors) == 0 or len(results) > 0,
        "num_examples": len(dataset),
        "num_successful": len([r for r in results.values() if "error" not in r]),
        "num_errors": len(errors),
        "errors": errors[:10],  # Limit error list
        "metrics": metrics,
        "results": results,
        "has_ollama": has_ollama,
    }


# =============================================================================
# UNIFIED WORKLOAD RUNNER
# =============================================================================

def run_workload_benchmark(
    identifier: str | int,
    iterations: Optional[int] = None,
    warmup: Optional[int] = None,
    run_langgraph: bool = True,
    run_apxm: bool = True,
) -> Dict[str, Any]:
    """
    Unified entry point for running any workload benchmark.

    Args:
        identifier: Workload number (1-10), name, or directory
        iterations: Number of benchmark iterations (default from workload config)
        warmup: Number of warmup iterations (default from workload config)
        run_langgraph: Whether to run LangGraph benchmark
        run_apxm: Whether to run A-PXM benchmark

    Returns:
        Dictionary with benchmark results
    """
    workload = get_workload(identifier)
    if workload is None:
        return {"error": f"Unknown workload: {identifier}"}

    iterations = iterations or workload.default_iterations
    warmup = warmup or workload.default_warmup

    results = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "workload": workload.name,
            "directory": workload.directory,
            "description": workload.description,
        },
        "config": {
            "iterations": iterations,
            "warmup": warmup,
            "workload_type": workload.workload_type.value,
        },
        "results": {},
    }

    # Select handlers based on workload type
    handlers = {
        WorkloadType.STANDARD: (_run_standard_langgraph, _run_standard_apxm),
        WorkloadType.ERROR_DETECTION: (_run_error_detection_langgraph, _run_error_detection_apxm),
        WorkloadType.SCALABILITY: (_run_scalability_langgraph, _run_scalability_apxm),
        WorkloadType.COMPILE_ONLY: (_run_compile_only_langgraph, _run_compile_only_apxm),
        WorkloadType.FUSION_COMPARISON: (_run_standard_langgraph, _run_standard_apxm),  # Use standard for now
        WorkloadType.LLM_PROBE: (_run_standard_langgraph, _run_standard_apxm),
        WorkloadType.TOKEN_ESTIMATION: (_run_compile_only_langgraph, _run_compile_only_apxm),
        WorkloadType.DATASET: (_run_dataset_langgraph, _run_dataset_apxm),
    }

    lg_handler, apxm_handler = handlers.get(workload.workload_type, (_run_standard_langgraph, _run_standard_apxm))

    # Run LangGraph benchmark
    if run_langgraph:
        try:
            results["results"]["langgraph"] = lg_handler(workload, iterations, warmup)
        except Exception as e:
            results["results"]["langgraph"] = {"error": str(e)}

    # Run A-PXM benchmark
    if run_apxm:
        try:
            results["results"]["apxm"] = apxm_handler(workload, iterations, warmup)
        except Exception as e:
            results["results"]["apxm"] = {"error": str(e)}

    return results


def list_workloads() -> List[Dict[str, Any]]:
    """List all available workloads."""
    return [
        {
            "number": num,
            "name": w.name,
            "directory": w.directory,
            "description": w.description,
            "type": w.workload_type.value,
        }
        for num, w in WORKLOADS.items()
    ]


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python apxm_runner.py <workflow.ais|workload_name> [iterations]")
        print("\nAvailable workloads:")
        for w in list_workloads():
            print(f"  {w['number']:2d}. {w['name']}: {w['description']}")
        sys.exit(1)

    arg = sys.argv[1]
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    # Check if it's a workload name or a .ais file
    if arg.endswith(".ais"):
        workflow = Path(arg)
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
    else:
        # Run workload benchmark
        results = run_workload_benchmark(arg, iterations)
        print(json.dumps(results, indent=2))
