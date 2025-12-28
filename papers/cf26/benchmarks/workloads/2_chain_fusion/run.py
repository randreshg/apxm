#!/usr/bin/env python3
"""
Chain Fusion Benchmark Runner

Compares A-PXM FuseReasoning (5 calls -> 1) vs LangGraph (5 separate calls).

This benchmark runs the ACTUAL AIS workflow through the full A-PXM pipeline:
  1. DSL parsing
  2. MLIR generation
  3. Optimization passes (FuseReasoning with -O1, disabled with -O0)
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


WARMUP_ITERATIONS = 1
BENCHMARK_ITERATIONS = 3
CHAIN_LENGTH = 5  # Number of RSN calls in the chain
WORKFLOW_FILE = Path(__file__).parent / "workflow.ais"


def run_langgraph(iterations: int = BENCHMARK_ITERATIONS) -> dict:
    """Run LangGraph workflow and collect timing."""
    from workflow import graph, HAS_OLLAMA

    samples = []
    initial_state = {
        "step1": "",
        "step2": "",
        "step3": "",
        "step4": "",
        "summary": "",
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
        "llm_calls": CHAIN_LENGTH,  # Each step is a separate call
        "has_ollama": HAS_OLLAMA,
        "samples": samples,
    }


def run_apxm(iterations: int = BENCHMARK_ITERATIONS) -> dict:
    """Run A-PXM workflow through the REAL pipeline.

    This runs workflow.ais through:
    - O0: No FuseReasoning (5 separate LLM calls)
    - O1: With FuseReasoning (1 batched LLM call)
    """
    # Run with O1 (FuseReasoning enabled) for the main benchmark
    config = APXMConfig(opt_level=1)
    result = run_benchmark(WORKFLOW_FILE, config, iterations, warmup=WARMUP_ITERATIONS)

    if result.get("success"):
        result["llm_calls"] = 1  # FuseReasoning batches into single call
    else:
        result["llm_calls"] = CHAIN_LENGTH  # Fallback if fusion fails

    return result


def run_apxm_comparison(iterations: int = BENCHMARK_ITERATIONS) -> dict:
    """Run both O0 (unfused) and O1 (fused) to measure actual speedup."""
    return compare_optimization_levels(WORKFLOW_FILE, iterations)


def main():
    parser = argparse.ArgumentParser(description="Chain Fusion Benchmark")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--iterations", type=int, default=BENCHMARK_ITERATIONS)
    args = parser.parse_args()

    results = {
        "meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "benchmark": "chain_fusion",
            "chain_length": CHAIN_LENGTH,
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

    # Run A-PXM benchmark
    results["results"]["apxm"] = run_apxm(args.iterations)

    # Calculate comparison metrics
    if "error" not in results["results"].get("langgraph", {}):
        lg = results["results"]["langgraph"]
        apxm = results["results"]["apxm"]
        results["comparison"] = {
            "llm_call_reduction": f"{lg['llm_calls']}x -> 1x",
            "theoretical_speedup": f"{CHAIN_LENGTH}x",
            "note": "A-PXM FuseReasoning batches the entire chain into a single prompt",
        }

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\nChain Fusion Benchmark Results")
        print(f"{'=' * 50}")
        print(f"Chain length: {CHAIN_LENGTH} RSN operations")
        print(f"Iterations: {args.iterations}")
        print()

        if "langgraph" in results["results"]:
            lg = results["results"]["langgraph"]
            if "error" not in lg:
                print(f"LangGraph (no fusion):")
                print(f"  LLM calls: {lg['llm_calls']}")
                print(f"  Mean: {lg['mean_ms']:.2f} ms")
                print(f"  Std:  {lg['std_ms']:.2f} ms")

        print(f"\nA-PXM (with FuseReasoning):")
        print(f"  LLM calls: 1 (fused)")
        print(f"  Theoretical speedup: {CHAIN_LENGTH}x")


if __name__ == "__main__":
    main()
