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

    # Build command
    cmd = [
        str(cli_path),
        "run",
        str(workflow_path),
        f"-O{config.opt_level}",
    ]
    metrics_path: Optional[Path] = None
    if capture_metrics:
        tmp = tempfile.NamedTemporaryFile(prefix="apxm_metrics_", suffix=".json", delete=False)
        metrics_path = Path(tmp.name)
        tmp.close()
        cmd.extend(["--emit-metrics", str(metrics_path)])

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
                    llm_requests = int(llm_metrics.get("total_requests", 0))
                    avg_latency_ms = float(llm_metrics.get("avg_latency_ms", 0))
                    llm_total_latency_ms = avg_latency_ms * llm_requests
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

    compiler_diagnostics = None
    if _emit_diagnostics():
        compiler_diagnostics = _run_compile_diagnostics(workflow_path, config.opt_level)

    # Warmup
    for _ in range(warmup):
        run_ais_workflow(workflow_path, config, capture_metrics=False)

    # Benchmark
    samples = []
    metrics_samples: List[Dict[str, float]] = []
    runtime_ms_values: List[float] = []
    compile_ms_values: List[float] = []
    artifact_ms_values: List[float] = []
    validation_ms_values: List[float] = []
    llm_total_ms_values: List[float] = []
    llm_requests_values: List[float] = []
    runtime_non_llm_ms_values: List[float] = []
    errors = []

    for i in range(iterations):
        result = run_ais_workflow(workflow_path, config)
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
                })
        else:
            errors.append(result.error)

    if not samples:
        return {
            "success": False,
            "error": errors[0] if errors else "No successful runs",
            "samples": [],
            "compiler": compiler_diagnostics,
        }

    import statistics

    def _stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {}
        return {
            "mean_ms": statistics.mean(values),
            "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min_ms": min(values),
            "max_ms": max(values),
            "p50_ms": statistics.median(values),
        }

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
        "compiler": compiler_diagnostics,
        "metrics": {
            "runtime_ms": _stats(runtime_ms_values),
            "compile_ms": _stats(compile_ms_values),
            "artifact_ms": _stats(artifact_ms_values),
            "validation_ms": _stats(validation_ms_values),
            "llm_total_ms": _stats(llm_total_ms_values),
            "llm_requests": _stats(llm_requests_values),
            "runtime_non_llm_ms": _stats(runtime_non_llm_ms_values),
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
    # For scalability workloads
    parallel_levels: List[int] = field(default_factory=list)
    # For compilation scaling
    op_counts: List[int] = field(default_factory=list)
    # Extra workflow files (for scalability, fusion quality, etc.)
    extra_workflows: Dict[str, str] = field(default_factory=dict)


# Workload registry - all 14 workloads
WORKLOADS: Dict[int, WorkloadConfig] = {
    1: WorkloadConfig(
        name="parallel_research",
        directory="1_parallel_research",
        description="Automatic parallelism from dataflow",
        initial_state={"topic": "quantum computing", "background": "", "advances": "", "impact": "", "combined": "", "report": ""},
    ),
    2: WorkloadConfig(
        name="chain_fusion",
        directory="2_chain_fusion",
        description="FuseAskOps compiler optimization (5 calls -> 1)",
        workload_type=WorkloadType.FUSION_COMPARISON,
        initial_state={"step1": "", "step2": "", "step3": "", "step4": "", "summary": ""},
    ),
    3: WorkloadConfig(
        name="type_verification",
        directory="3_type_verification",
        description="Compile-time vs runtime error detection",
        workload_type=WorkloadType.ERROR_DETECTION,
        initial_state={"result": "", "output": ""},
    ),
    4: WorkloadConfig(
        name="scalability",
        directory="4_scalability",
        description="N-way parallelism efficiency (N=2,4,8)",
        workload_type=WorkloadType.SCALABILITY,
        parallel_levels=[2, 4, 8],
        extra_workflows={"2": "workflow_n2.ais", "4": "workflow_n4.ais", "8": "workflow_n8.ais"},
    ),
    5: WorkloadConfig(
        name="memory_augmented",
        directory="5_memory_augmented",
        description="3-tier memory system (STM/LTM/Episodic)",
        initial_state={"query": "What is machine learning?", "cached": "", "answer": ""},
    ),
    6: WorkloadConfig(
        name="tool_invocation",
        directory="6_tool_invocation",
        description="Native INV operations with capability system",
        initial_state={"query": "Search for recent AI news", "tool_result": "", "answer": ""},
    ),
    7: WorkloadConfig(
        name="reflection",
        directory="7_reflection",
        description="Built-in reflect operation for self-improvement",
        initial_state={"task": "Write a haiku about coding", "initial_answer": "", "reflection": "", "improved_answer": ""},
    ),
    8: WorkloadConfig(
        name="planning",
        directory="8_planning",
        description="Native plan operation for task decomposition",
        initial_state={"goal": "Build a simple web app", "steps": "", "step1_result": "", "step2_result": "", "step3_result": "", "combined": "", "final_result": ""},
    ),
    9: WorkloadConfig(
        name="conditional_routing",
        directory="9_conditional_routing",
        description="Dataflow-based parallel preparation and routing",
        initial_state={"input": "Explain quantum entanglement", "category": "", "output": ""},
    ),
    10: WorkloadConfig(
        name="multi_agent",
        directory="10_multi_agent",
        description="Native agent definitions with communicate operations",
        initial_state={"topic": "climate change", "research_result": "", "critique_prep": "", "critique_result": "", "final_report": ""},
    ),
    11: WorkloadConfig(
        name="compilation_scaling",
        directory="11_compilation_scaling",
        description="Compilation phase timing at different op counts",
        workload_type=WorkloadType.COMPILE_ONLY,
        op_counts=[10, 25, 50, 100],
    ),
    12: WorkloadConfig(
        name="real_llm_probe",
        directory="12_real_llm_probe",
        description="Real LLM latency and token measurement",
        workload_type=WorkloadType.LLM_PROBE,
    ),
    13: WorkloadConfig(
        name="fusion_quality",
        directory="13_fusion_quality",
        description="FuseAskOps optimization effectiveness by task type",
        workload_type=WorkloadType.FUSION_COMPARISON,
        extra_workflows={
            "classification": "classification.ais",
            "extraction": "extraction.ais",
            "reasoning": "reasoning.ais",
            "creative": "creative.ais",
        },
    ),
    14: WorkloadConfig(
        name="token_estimation",
        directory="14_token_estimation",
        description="Token cost estimation before/after fusion",
        workload_type=WorkloadType.TOKEN_ESTIMATION,
        extra_workflows={
            "simple_chain": "simple_chain.ais",
            "sequential_reasoning": "sequential_reasoning.ais",
            "parallel_research": "parallel_research.ais",
        },
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
    return run_benchmark(workflow_file, config, iterations, warmup=warmup)


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
    result = subprocess.run(
        [str(cli_path), "run", str(workflow_file), "-O0"],
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
            lines.append(f'        rsn "Task {i}: Process data element" -> {result_name}')
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
        identifier: Workload number (1-14), name, or directory
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
