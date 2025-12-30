#!/usr/bin/env python3
"""
Planning Benchmark Runner

Compares A-PXM's native PLAN operation vs LangGraph's CoT prompting.

This benchmark runs the ACTUAL AIS workflow through the full A-PXM pipeline.
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
GOAL = "Build a simple web application"
WORKFLOW_FILE = Path(__file__).parent / "workflow.ais"


def run_langgraph_planning(iterations: int = BENCHMARK_ITERATIONS, warmup: int = WARMUP_ITERATIONS) -> dict:
    """Run LangGraph planning workflow and measure timing."""
    from workflow import graph, HAS_OLLAMA

    initial_state = {
        "goal": GOAL,
        "steps": [],
        "step_results": [],
        "final_result": "",
    }

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

    return {
        "mean_ms": statistics.mean(samples),
        "std_ms": statistics.stdev(samples) if len(samples) > 1 else 0,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "p50_ms": statistics.median(samples),
        "has_ollama": HAS_OLLAMA,
        "llm_calls": 5,  # plan + 3 execute + synthesize
        "samples": samples,
    }


def run_apxm_planning(iterations: int = BENCHMARK_ITERATIONS, warmup: int = WARMUP_ITERATIONS) -> dict:
    """Run A-PXM planning workflow through the REAL pipeline."""
    config = APXMConfig(opt_level=1)
    return run_benchmark(WORKFLOW_FILE, config, iterations, warmup=warmup)


def run_langgraph(iterations: int = BENCHMARK_ITERATIONS, warmup: int = WARMUP_ITERATIONS) -> dict:
    """Entry point for the suite runner (`runner.py`)."""
    return run_langgraph_planning(iterations, warmup=warmup)


def run_apxm(iterations: int = BENCHMARK_ITERATIONS, warmup: int = WARMUP_ITERATIONS) -> dict:
    """Entry point for the suite runner (`runner.py`)."""
    return run_apxm_planning(iterations, warmup=warmup)


def main():
    parser = argparse.ArgumentParser(description="Planning Benchmark")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--iterations", type=int, default=BENCHMARK_ITERATIONS)
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERATIONS)
    args = parser.parse_args()

    results = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmark": "planning",
            "goal": GOAL,
        },
        "config": {
            "iterations": args.iterations,
            "warmup": args.warmup,
        },
        "results": {},
    }

    # Run LangGraph benchmark
    try:
        results["results"]["langgraph"] = run_langgraph_planning(args.iterations, warmup=args.warmup)
    except ImportError as e:
        results["results"]["langgraph"] = {"error": str(e)}

    # Run A-PXM benchmark
    results["results"]["apxm"] = run_apxm_planning(args.iterations, warmup=args.warmup)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\nPlanning Benchmark Results")
        print(f"{'=' * 50}")
        print(f"Goal: {GOAL}")
        print(f"Iterations: {args.iterations}")
        print()

        if "langgraph" in results["results"]:
            lg = results["results"]["langgraph"]
            if "error" not in lg:
                print(f"LangGraph:")
                print(f"  Mean: {lg['mean_ms']:.2f} ms")
                print(f"  Std:  {lg['std_ms']:.2f} ms")
                print(f"  LLM calls: {lg.get('llm_calls', 'N/A')}")
                print(f"  Has Ollama: {lg.get('has_ollama', False)}")

        print()
        apxm = results["results"].get("apxm", {})
        print(f"A-PXM (with PLAN operation):")
        if apxm.get("success"):
            print(f"  Mean: {apxm['mean_ms']:.2f} ms")
            print(f"  Std:  {apxm.get('std_ms', 0):.2f} ms")
        elif apxm.get("error"):
            print(f"  Error: {apxm['error']}")


if __name__ == "__main__":
    main()
