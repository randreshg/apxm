#!/usr/bin/env python3
"""
Export benchmark results JSON into a plot-ready CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


COLUMNS = [
    "source_file",
    "benchmark_suite",
    "timestamp",
    "workload",
    "system",
    "status",
    "error",
    "iterations",
    "warmup",
    # Wall time metrics
    "wall_ms_mean",
    "wall_ms_std",
    "wall_ms_p50",
    "wall_ms_min",
    "wall_ms_max",
    # Runtime breakdown
    "runtime_ms_mean",
    "compile_total_ms",
    "artifact_gen_ms",
    "validation_ms_mean",
    # LLM metrics
    "llm_total_ms_mean",
    "llm_wall_ms_mean",
    "non_llm_ms_mean",
    "llm_requests_mean",
    "llm_calls_reported",
    # Framework overhead (compile + artifact + validation)
    "framework_overhead_ms",
    # Optimization level
    "opt_level",
    # DAG structure
    "passes_applied",
    "dag_nodes",
    "dag_edges",
    "dag_entry_nodes",
    # Work-Span metrics (execution model properties)
    "T_1",  # Total work (LLM calls)
    "T_inf",  # Critical path depth (span)
    "theoretical_speedup",  # T_1 / T_inf
    # Normalized latency (constant LLM delay = 1000ms)
    "normalized_latency_ms",  # T_inf * 1000ms (critical path latency)
    # LLM backend
    "has_ollama",
]

# Constant delay for normalized comparison (1 second per LLM call)
CONSTANT_LLM_DELAY_MS = 1000.0


def get_nested(data: Dict[str, Any], path: Iterable[str]) -> Optional[Any]:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
        if cur is None:
            return None
    return cur


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def min_or_none(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def max_or_zero(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return max(value, 0.0)


def summarize_langgraph(entry: Dict[str, Any]) -> Dict[str, Optional[float]]:
    llm = entry.get("llm", {}) if isinstance(entry.get("llm"), dict) else {}
    wall = to_float(entry.get("mean_ms"))
    llm_total = to_float(llm.get("total_ms_mean"))
    llm_wall = to_float(llm.get("wall_llm_ms_mean"))
    if llm_wall is None and wall is not None and llm_total is not None:
        llm_wall = min_or_none(llm_total, wall)
    non_llm = to_float(llm.get("non_llm_ms_mean"))
    if non_llm is None and wall is not None and llm_wall is not None:
        non_llm = max_or_zero(wall - llm_wall)

    # Work-Span: LangGraph is sequential, so T_inf = T_1
    llm_calls = to_float(llm.get("calls_mean"))
    T_1 = llm_calls
    T_inf = llm_calls  # Sequential execution
    theoretical_speedup = 1.0 if T_inf else None  # No parallelism

    # Normalized latency = T_inf * constant delay (framework-agnostic)
    normalized_latency = T_inf * CONSTANT_LLM_DELAY_MS if T_inf else None

    return {
        "wall_ms_mean": wall,
        "wall_ms_std": to_float(entry.get("std_ms")),
        "wall_ms_p50": to_float(entry.get("p50_ms")),
        "wall_ms_min": to_float(entry.get("min_ms")),
        "wall_ms_max": to_float(entry.get("max_ms")),
        "llm_total_ms_mean": llm_total,
        "llm_wall_ms_mean": llm_wall,
        "non_llm_ms_mean": non_llm,
        "llm_requests_mean": llm_calls,
        "framework_overhead_ms": 0.0,  # LangGraph has negligible framework overhead
        "T_1": T_1,
        "T_inf": T_inf,
        "theoretical_speedup": theoretical_speedup,
        "normalized_latency_ms": normalized_latency,
    }


def compute_critical_path(
    llm_calls: Optional[float], entry_nodes: Optional[float], exit_nodes: Optional[float]
) -> Optional[float]:
    """
    Estimate critical path depth (T_inf) from DAG statistics.

    For a fan-out/fan-in pattern (N parallel -> merge -> M sequential):
      The critical path is the longest path through the DAG.

    Workload 1 example (3 parallel research + 1 synthesizer):
      entry_nodes=4 (includes the 3 research + 1 trigger), llm_calls=4
      Actual structure: 3 parallel RSN -> merge -> 1 sequential RSN
      Critical path: 1 (parallel layer) + 1 (sequential) = 2

    The key insight: if exit_nodes=1 (single sink), there's likely a merge
    before a final sequential step, giving depth = 2 for fan-out/fan-in.
    """
    if llm_calls is None or llm_calls <= 0:
        return None
    if entry_nodes is None or entry_nodes <= 1:
        return llm_calls  # No parallelism, sequential execution

    # For fan-out/fan-in pattern with single exit: depth = 2
    # (parallel layer + merge/sequential layer)
    if exit_nodes == 1 and entry_nodes > 1:
        # Classic fan-out/fan-in: N parallel inputs -> merge -> 1 output
        # Critical path = 2 (one parallel level, one sequential after merge)
        # Unless all calls are purely parallel (no sequential after merge)
        if llm_calls > 1:
            return 2.0

    # For other patterns, estimate based on parallelism degree
    parallel_width = min(entry_nodes, llm_calls)
    if parallel_width >= llm_calls:
        return 1.0  # All parallel, single depth

    # General case: ceil(llm_calls / parallel_width)
    return max(2.0, math.ceil(llm_calls / parallel_width))


def summarize_apxm(entry: Dict[str, Any]) -> Dict[str, Optional[float]]:
    metrics = entry.get("metrics", {}) if isinstance(entry.get("metrics"), dict) else {}
    runtime_mean = to_float(get_nested(metrics, ["runtime_ms", "mean_ms"]))
    llm_total = to_float(get_nested(metrics, ["llm_total_ms", "mean_ms"]))
    llm_wall = min_or_none(llm_total, runtime_mean)
    non_llm = to_float(get_nested(metrics, ["runtime_non_llm_ms", "mean_ms"]))
    if non_llm is None and runtime_mean is not None and llm_wall is not None:
        non_llm = max_or_zero(runtime_mean - llm_wall)

    compiler_diag = get_nested(entry, ["compiler", "diagnostics"]) or {}
    if not isinstance(compiler_diag, dict):
        compiler_diag = {}

    compilation_phases = compiler_diag.get("compilation_phases", {})
    dag_stats = compiler_diag.get("dag_statistics", {})

    # Extract metrics
    compile_total = to_float(compilation_phases.get("total_ms"))
    artifact_gen = to_float(compilation_phases.get("artifact_gen_ms"))
    validation_mean = to_float(get_nested(metrics, ["validation_ms", "mean_ms"]))
    llm_calls = to_float(get_nested(metrics, ["llm_requests", "mean_ms"]))
    entry_nodes = to_float(dag_stats.get("entry_nodes"))
    exit_nodes = to_float(dag_stats.get("exit_nodes"))
    opt_level = entry.get("opt_level")

    # Compute framework overhead = compile + artifact + validation
    overhead_parts = [compile_total, artifact_gen, validation_mean]
    framework_overhead = sum(p for p in overhead_parts if p is not None) or None

    # Work-Span metrics
    T_1 = llm_calls  # Total work (all LLM calls)
    T_inf = compute_critical_path(llm_calls, entry_nodes, exit_nodes)
    theoretical_speedup = None
    if T_1 is not None and T_inf is not None and T_inf > 0:
        theoretical_speedup = T_1 / T_inf

    # Normalized latency = T_inf * constant delay (framework-agnostic)
    normalized_latency = T_inf * CONSTANT_LLM_DELAY_MS if T_inf else None

    return {
        "wall_ms_mean": to_float(entry.get("mean_ms")),
        "wall_ms_std": to_float(entry.get("std_ms")),
        "wall_ms_p50": to_float(entry.get("p50_ms")),
        "wall_ms_min": to_float(entry.get("min_ms")),
        "wall_ms_max": to_float(entry.get("max_ms")),
        "runtime_ms_mean": runtime_mean,
        "compile_total_ms": compile_total,
        "artifact_gen_ms": artifact_gen,
        "validation_ms_mean": validation_mean,
        "llm_total_ms_mean": llm_total,
        "llm_wall_ms_mean": llm_wall,
        "non_llm_ms_mean": non_llm,
        "llm_requests_mean": llm_calls,
        "framework_overhead_ms": framework_overhead,
        "opt_level": opt_level,
        "passes_applied": ";".join(compiler_diag.get("passes_applied", []))
        if isinstance(compiler_diag.get("passes_applied"), list)
        else None,
        "dag_nodes": to_float(dag_stats.get("total_nodes")),
        "dag_edges": to_float(dag_stats.get("total_edges")),
        "dag_entry_nodes": entry_nodes,
        "T_1": T_1,
        "T_inf": T_inf,
        "theoretical_speedup": theoretical_speedup,
        "normalized_latency_ms": normalized_latency,
    }


def build_row(
    source_file: Path,
    meta: Dict[str, Any],
    config: Dict[str, Any],
    workload_name: str,
    system: str,
    entry: Dict[str, Any],
) -> Dict[str, Any]:
    status = "ok"
    error = None

    if entry is None:
        status = "error"
        error = "missing"
        entry = {}
    elif entry.get("error"):
        status = "error"
        error = entry.get("error")
    elif system == "apxm" and entry.get("success") is False:
        status = "error"
        error = entry.get("error")

    row: Dict[str, Any] = {
        "source_file": str(source_file),
        "benchmark_suite": meta.get("benchmark_suite"),
        "timestamp": meta.get("timestamp"),
        "workload": workload_name,
        "system": system,
        "status": status,
        "error": error,
        "iterations": config.get("iterations"),
        "warmup": config.get("warmup"),
        "llm_calls_reported": entry.get("llm_calls"),
        "has_ollama": entry.get("has_ollama"),
    }

    metrics: Dict[str, Optional[float]] = {}
    if status == "ok":
        if system == "langgraph":
            metrics = summarize_langgraph(entry)
        elif system == "apxm":
            metrics = summarize_apxm(entry)

    row.update(metrics)
    return row


def export_file(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    meta = data.get("meta", {})
    config = data.get("config", {})
    workloads = data.get("workloads", {})

    rows: List[Dict[str, Any]] = []
    for workload_name, result in workloads.items():
        if not isinstance(result, dict):
            continue
        for system in ("langgraph", "apxm"):
            entry = result.get(system, {})
            row = build_row(path, meta, config, workload_name, system, entry)
            rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in COLUMNS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Export benchmark JSON to CSV")
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON files")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path",
    )
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for path in args.inputs:
        rows.extend(export_file(path))

    write_csv(rows, args.output)


if __name__ == "__main__":
    main()
