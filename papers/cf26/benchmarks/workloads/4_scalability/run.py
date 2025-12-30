#!/usr/bin/env python3
"""
Scalability Benchmark Runner

Measures parallelism efficiency at N = 2, 4, 8 concurrent operations.

This benchmark runs the ACTUAL AIS workflow through the full A-PXM pipeline.
Note: The workflow.ais contains multiple flows (parallel_2, parallel_4, parallel_8).
Currently runs the default main flow; specific flow selection requires CLI enhancement.
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to import apxm_runner
sys.path.insert(0, str(Path(__file__).parent.parent))
from apxm_runner import APXMConfig, run_benchmark

def _get_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


WARMUP_ITERATIONS = _get_int_env("APXM_BENCH_WARMUP", 3)
BENCHMARK_ITERATIONS = _get_int_env("APXM_BENCH_ITERATIONS", 10)
PARALLEL_LEVELS = [2, 4, 8]
SIMULATED_DELAY_MS = 100
WORKFLOW_FILE = Path(__file__).parent / "workflow.ais"


def run_langgraph_scalability(n: int, iterations: int, warmup: int = WARMUP_ITERATIONS) -> dict:
    """Run LangGraph scalability test for N parallel ops."""
    from workflow import build_parallel_graph, HAS_OLLAMA

    graph = build_parallel_graph(n)
    initial_state = {"results": [], "final": ""}

    samples = []

    # Warmup
    for _ in range(warmup):
        graph.invoke(initial_state)

    # Benchmark
    for _ in range(iterations):
        start = time.perf_counter()
        graph.invoke(initial_state)
        elapsed_ms = (time.perf_counter() - start) * 1000
        samples.append(elapsed_ms)

    mean_ms = statistics.mean(samples)
    sequential_time = n * SIMULATED_DELAY_MS
    speedup = sequential_time / mean_ms
    efficiency = speedup / n * 100

    return {
        "n": n,
        "mean_ms": mean_ms,
        "std_ms": statistics.stdev(samples) if len(samples) > 1 else 0,
        "p50_ms": statistics.median(samples),
        "speedup": speedup,
        "efficiency_pct": efficiency,
        "theoretical_speedup": float(n),
        "has_ollama": HAS_OLLAMA,
        "samples": samples,
    }


def run_apxm_scalability(n: int, iterations: int, warmup: int = WARMUP_ITERATIONS) -> dict:
    """Run A-PXM scalability test through the REAL pipeline.

    Note: Currently runs the default flow. For specific N-way parallel testing,
    the CLI needs flow selection support (--flow parallel_N).
    """
    config = APXMConfig(opt_level=1)
    result = run_benchmark(WORKFLOW_FILE, config, iterations, warmup=warmup)

    if result.get("success"):
        mean_ms = result["mean_ms"]
        sequential_time = n * SIMULATED_DELAY_MS
        speedup = sequential_time / mean_ms if mean_ms > 0 else 0
        efficiency = (speedup / n * 100) if n > 0 else 0

        return {
            "n": n,
            "mean_ms": mean_ms,
            "std_ms": result.get("std_ms", 0),
            "p50_ms": result.get("p50_ms", mean_ms),
            "speedup": speedup,
            "efficiency_pct": efficiency,
            "theoretical_speedup": float(n),
            "success": True,
        }
    else:
        return {
            "n": n,
            "mean_ms": 0,
            "std_ms": 0,
            "p50_ms": 0,
            "speedup": 0,
            "efficiency_pct": 0,
            "theoretical_speedup": float(n),
            "error": result.get("error", "Unknown error"),
        }


def run_langgraph(iterations: int = BENCHMARK_ITERATIONS, warmup: int = WARMUP_ITERATIONS) -> dict:
    """Entry point for the suite runner (`runner.py`)."""
    series = [run_langgraph_scalability(n, iterations, warmup=warmup) for n in PARALLEL_LEVELS]
    return {"series": series, "parallel_levels": PARALLEL_LEVELS}


def run_apxm(iterations: int = BENCHMARK_ITERATIONS, warmup: int = WARMUP_ITERATIONS) -> dict:
    """Entry point for the suite runner (`runner.py`)."""
    series = [run_apxm_scalability(n, iterations, warmup=warmup) for n in PARALLEL_LEVELS]
    return {"series": series, "parallel_levels": PARALLEL_LEVELS}


def main():
    parser = argparse.ArgumentParser(description="Scalability Benchmark")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--iterations", type=int, default=BENCHMARK_ITERATIONS)
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERATIONS)
    args = parser.parse_args()

    results = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmark": "scalability",
            "simulated_delay_ms": SIMULATED_DELAY_MS,
        },
        "config": {
            "iterations": args.iterations,
            "warmup": args.warmup,
            "parallel_levels": PARALLEL_LEVELS,
        },
        "results": {
            "langgraph": [],
            "apxm": [],
        },
    }

    # Run benchmarks for each parallelism level
    for n in PARALLEL_LEVELS:
        try:
            lg_result = run_langgraph_scalability(n, args.iterations, warmup=args.warmup)
            results["results"]["langgraph"].append(lg_result)
        except ImportError as e:
            results["results"]["langgraph"].append({"n": n, "error": str(e)})

        apxm_result = run_apxm_scalability(n, args.iterations, warmup=args.warmup)
        results["results"]["apxm"].append(apxm_result)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\nScalability Benchmark Results")
        print(f"{'=' * 60}")
        print(f"Simulated LLM delay: {SIMULATED_DELAY_MS}ms")
        print(f"Iterations: {args.iterations}")
        print()

        print(f"{'N':>3} | {'Theoretical':>11} | {'LangGraph':>20} | {'A-PXM':>15}")
        print(f"{'-'*3}-+-{'-'*11}-+-{'-'*20}-+-{'-'*15}")

        lg_results = {r["n"]: r for r in results["results"]["langgraph"] if "error" not in r}
        apxm_results = {r["n"]: r for r in results["results"]["apxm"]}

        for n in PARALLEL_LEVELS:
            theoretical = f"{n:.2f}x"
            lg = lg_results.get(n, {})
            apxm = apxm_results.get(n, {})

            lg_str = f"{lg.get('speedup', 0):.2f}x ({lg.get('efficiency_pct', 0):.0f}%)" if lg else "N/A"
            if apxm.get("success"):
                apxm_str = f"{apxm.get('efficiency_pct', 0):.0f}%"
            elif apxm.get("error"):
                apxm_str = "Error"
            else:
                apxm_str = "N/A"

            print(f"{n:>3} | {theoretical:>11} | {lg_str:>20} | {apxm_str:>15}")


if __name__ == "__main__":
    main()
