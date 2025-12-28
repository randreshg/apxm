#!/usr/bin/env python3
"""
Multi-Agent Benchmark Runner

Compares A-PXM's native agent definitions and communicate operations
vs LangGraph's subgraph composition.

This benchmark runs the ACTUAL AIS workflow through the full A-PXM pipeline.
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
from apxm_runner import APXMConfig, run_benchmark

WARMUP_ITERATIONS = 1
BENCHMARK_ITERATIONS = 3
TOPIC = "The future of renewable energy"
WORKFLOW_FILE = Path(__file__).parent / "workflow.ais"


def run_langgraph_multiagent(iterations: int = BENCHMARK_ITERATIONS) -> dict:
    """Run LangGraph multi-agent workflow and measure timing."""
    from workflow import graph, HAS_OLLAMA

    initial_state = {
        "topic": TOPIC,
        "research_result": "",
        "critique_result": "",
        "final_report": "",
    }

    samples = []

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
        "agents": 3,  # researcher, critic, coordinator
        "llm_calls": 3,  # research + critique + synthesize
        "samples": samples,
    }


def run_apxm_multiagent(iterations: int = BENCHMARK_ITERATIONS) -> dict:
    """Run A-PXM multi-agent workflow through the REAL pipeline."""
    config = APXMConfig(opt_level=1)
    return run_benchmark(WORKFLOW_FILE, config, iterations, warmup=WARMUP_ITERATIONS)


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Benchmark")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--iterations", type=int, default=BENCHMARK_ITERATIONS)
    args = parser.parse_args()

    results = {
        "meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "benchmark": "multi_agent",
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
        results["results"]["langgraph"] = run_langgraph_multiagent(args.iterations)
    except ImportError as e:
        results["results"]["langgraph"] = {"error": str(e)}

    # Run A-PXM benchmark
    results["results"]["apxm"] = run_apxm_multiagent(args.iterations)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\nMulti-Agent Benchmark Results")
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
                print(f"  Agents: {lg.get('agents', 'N/A')}")
                print(f"  LLM calls: {lg.get('llm_calls', 'N/A')}")
                print(f"  Has Ollama: {lg.get('has_ollama', False)}")

        print()
        apxm = results["results"].get("apxm", {})
        print(f"A-PXM (with multi-agent):")
        if apxm.get("success"):
            print(f"  Mean: {apxm['mean_ms']:.2f} ms")
            print(f"  Std:  {apxm.get('std_ms', 0):.2f} ms")
        elif apxm.get("error"):
            print(f"  Error: {apxm['error']}")


if __name__ == "__main__":
    main()
