#!/usr/bin/env python3
"""
Parallel Research Benchmark Runner

Compares A-PXM automatic parallelism vs LangGraph explicit Send API.

This benchmark runs the ACTUAL AIS workflow through the full A-PXM pipeline:
  1. DSL parsing
  2. MLIR generation
  3. Optimization passes
  4. Artifact generation
  5. Runtime execution with real LLM calls
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to import apxm_runner
sys.path.insert(0, str(Path(__file__).parent.parent))
from apxm_runner import APXMConfig, run_benchmark, compare_optimization_levels

# Configuration
WARMUP_ITERATIONS = 1
BENCHMARK_ITERATIONS = 3
TOPIC = "quantum computing"
WORKFLOW_FILE = Path(__file__).parent / "workflow.ais"


def run_langgraph(iterations: int = BENCHMARK_ITERATIONS) -> dict:
    """Run LangGraph workflow and collect timing."""
    from workflow import graph, HAS_OLLAMA

    samples = []
    initial_state = {
        "topic": TOPIC,
        "background": "",
        "advances": "",
        "impact": "",
        "combined": "",
        "report": "",
    }

    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        graph.invoke(initial_state)

    # Benchmark
    for _ in range(iterations):
        start = time.perf_counter()
        graph.invoke(initial_state)
        elapsed_ms = (time.perf_counter() - start) * 1000
        samples.append(elapsed_ms)

    return {
        "mean_ms": statistics.mean(samples),
        "std_ms": statistics.stdev(samples) if len(samples) > 1 else 0,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "p50_ms": statistics.median(samples),
        "has_ollama": HAS_OLLAMA,
        "samples": samples,
    }


def run_apxm(iterations: int = BENCHMARK_ITERATIONS) -> dict:
    """Run A-PXM workflow through the REAL pipeline."""
    config = APXMConfig(opt_level=1)
    return run_benchmark(WORKFLOW_FILE, config, iterations, warmup=WARMUP_ITERATIONS)


def run_apxm_comparison(iterations: int = BENCHMARK_ITERATIONS) -> dict:
    """Run both O0 (unfused) and O1 (fused) to measure actual speedup."""
    return compare_optimization_levels(WORKFLOW_FILE, iterations)


def main():
    parser = argparse.ArgumentParser(description="Parallel Research Benchmark")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--iterations", type=int, default=BENCHMARK_ITERATIONS)
    parser.add_argument("--langgraph-only", action="store_true")
    args = parser.parse_args()

    results = {
        "meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "benchmark": "parallel_research",
            "topic": TOPIC,
        },
        "config": {
            "iterations": args.iterations,
            "warmup": WARMUP_ITERATIONS,
        },
        "results": {},
    }

    # Run LangGraph benchmark
    try:
        results["results"]["langgraph"] = run_langgraph(args.iterations)
    except ImportError as e:
        results["results"]["langgraph"] = {"error": str(e)}

    # Run A-PXM benchmark (if not langgraph-only)
    if not args.langgraph_only:
        results["results"]["apxm"] = run_apxm(args.iterations)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\nParallel Research Benchmark Results")
        print(f"{'=' * 50}")
        print(f"Topic: {TOPIC}")
        print(f"Iterations: {args.iterations}")
        print()

        if "langgraph" in results["results"]:
            lg = results["results"]["langgraph"]
            if "error" not in lg:
                print(f"LangGraph:")
                print(f"  Mean: {lg['mean_ms']:.2f} ms")
                print(f"  Std:  {lg['std_ms']:.2f} ms")
                print(f"  P50:  {lg['p50_ms']:.2f} ms")
            else:
                print(f"LangGraph: Error - {lg['error']}")

        if "apxm" in results["results"]:
            apxm = results["results"]["apxm"]
            if apxm.get("success"):
                print(f"\nA-PXM (with parallelism):")
                print(f"  Mean: {apxm['mean_ms']:.2f} ms")
                print(f"  Std:  {apxm.get('std_ms', 0):.2f} ms")
                print(f"  P50:  {apxm.get('p50_ms', apxm['mean_ms']):.2f} ms")
            elif apxm.get("error"):
                print(f"\nA-PXM: Error - {apxm['error']}")


if __name__ == "__main__":
    main()
