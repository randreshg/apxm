#!/usr/bin/env python3
"""
Memory Augmented Benchmark Runner

Compares A-PXM's native qmem/umem memory operations vs LangGraph's checkpoint approach.
A-PXM supports 3-tier memory: STM (short-term), LTM (long-term), and Episodic.

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
QUERY = "What is quantum computing?"
WORKFLOW_FILE = Path(__file__).parent / "workflow.ais"


def run_langgraph_memory(iterations: int = BENCHMARK_ITERATIONS, warmup: int = WARMUP_ITERATIONS) -> dict:
    """Run LangGraph memory workflow and measure timing."""
    from workflow import graph, HAS_CHECKPOINT

    initial_state = {
        "query": QUERY,
        "stm": {},
        "ltm": {"domain_knowledge": "Quantum computing fundamentals"},
        "episodic": [],
        "cached": "",
        "answer": "",
    }

    config = {}
    if HAS_CHECKPOINT:
        config = {"configurable": {"thread_id": "benchmark"}}

    samples = []

    # Warmup
    for i in range(warmup):
        if HAS_CHECKPOINT:
            config["configurable"]["thread_id"] = f"warmup_{i}"
        graph.invoke(initial_state, config)

    # Benchmark
    for i in range(iterations):
        if HAS_CHECKPOINT:
            config["configurable"]["thread_id"] = f"bench_{i}"

        start = time.perf_counter()
        graph.invoke(initial_state, config)
        elapsed_ms = (time.perf_counter() - start) * 1000
        samples.append(elapsed_ms)

    return {
        "mean_ms": statistics.mean(samples),
        "std_ms": statistics.stdev(samples) if len(samples) > 1 else 0,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "p50_ms": statistics.median(samples),
        "has_checkpoint": HAS_CHECKPOINT,
        "samples": samples,
    }


def run_apxm_memory(iterations: int = BENCHMARK_ITERATIONS, warmup: int = WARMUP_ITERATIONS) -> dict:
    """Run A-PXM memory workflow through the REAL pipeline."""
    config = APXMConfig(opt_level=1)
    return run_benchmark(WORKFLOW_FILE, config, iterations, warmup=warmup)


def run_langgraph(iterations: int = BENCHMARK_ITERATIONS, warmup: int = WARMUP_ITERATIONS) -> dict:
    """Entry point for the suite runner (`runner.py`)."""
    return run_langgraph_memory(iterations, warmup=warmup)


def run_apxm(iterations: int = BENCHMARK_ITERATIONS, warmup: int = WARMUP_ITERATIONS) -> dict:
    """Entry point for the suite runner (`runner.py`)."""
    return run_apxm_memory(iterations, warmup=warmup)


def main():
    parser = argparse.ArgumentParser(description="Memory Augmented Benchmark")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--iterations", type=int, default=BENCHMARK_ITERATIONS)
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERATIONS)
    args = parser.parse_args()

    results = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmark": "memory_augmented",
            "query": QUERY,
        },
        "config": {
            "iterations": args.iterations,
            "warmup": args.warmup,
        },
        "results": {},
    }

    # Run LangGraph benchmark
    try:
        results["results"]["langgraph"] = run_langgraph_memory(args.iterations, warmup=args.warmup)
    except ImportError as e:
        results["results"]["langgraph"] = {"error": str(e)}

    # Run A-PXM benchmark
    results["results"]["apxm"] = run_apxm_memory(args.iterations, warmup=args.warmup)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\nMemory Augmented Benchmark Results")
        print(f"{'=' * 50}")
        print(f"Query: {QUERY}")
        print(f"Iterations: {args.iterations}")
        print()

        if "langgraph" in results["results"]:
            lg = results["results"]["langgraph"]
            if "error" not in lg:
                print(f"LangGraph:")
                print(f"  Mean: {lg['mean_ms']:.2f} ms")
                print(f"  Std:  {lg['std_ms']:.2f} ms")
                print(f"  Has checkpoint: {lg.get('has_checkpoint', False)}")

        print()
        apxm = results["results"].get("apxm", {})
        print(f"A-PXM (with native memory):")
        if apxm.get("success"):
            print(f"  Mean: {apxm['mean_ms']:.2f} ms")
            print(f"  Std:  {apxm.get('std_ms', 0):.2f} ms")
        elif apxm.get("error"):
            print(f"  Error: {apxm['error']}")


if __name__ == "__main__":
    main()
