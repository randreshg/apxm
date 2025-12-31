#!/usr/bin/env python3
"""
Master Benchmark Runner

Runs all DSL comparison benchmarks using the consolidated apxm_runner.
This is a thin wrapper around apxm_runner.run_workload_benchmark().

Usage:
    python runner.py --json                    # Run all, output JSON
    python runner.py --workload 1              # Run specific workload
    python runner.py --list                    # List available workloads
    python runner.py --output results.json     # Save to file
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# Import consolidated runner
from apxm_runner import (
    WORKLOADS,
    list_workloads,
    run_workload_benchmark,
)


def get_platform_info() -> dict:
    """Collect platform information."""
    return {
        "os": platform.system(),
        "os_version": platform.release(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="A-PXM vs LangGraph DSL Comparison Benchmark Runner"
    )
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--list", action="store_true", help="List available workloads")
    parser.add_argument("--workload", type=int, help="Run specific workload (1-14)")
    parser.add_argument("--iterations", type=int, default=10, help="Iterations per benchmark")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations (not counted)")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--apxm-only", action="store_true", help="Run only A-PXM benchmarks")
    parser.add_argument("--langgraph-only", action="store_true", help="Run only LangGraph benchmarks")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable Workloads:")
        print("=" * 60)
        for w in list_workloads():
            print(f"  {w['number']:2d}. {w['name']}")
            print(f"     {w['description']}")
            print(f"     Type: {w['type']}")
            print()
        return

    # Build full results
    results = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmark_suite": "apxm_dsl_comparison",
            "version": "3.0",
            "platform": get_platform_info(),
        },
        "config": {
            "iterations": args.iterations,
            "warmup": args.warmup,
            "note": "Warmup iterations are executed but discarded from results",
        },
        "workloads": {},
    }

    # Determine which benchmarks to run
    run_lg = not args.apxm_only
    run_apxm = not args.langgraph_only

    # Run specified workload or all
    workloads_to_run = [args.workload] if args.workload else list(WORKLOADS.keys())

    for num in workloads_to_run:
        workload = WORKLOADS.get(num)
        if not workload:
            continue

        if not args.json:
            print(f"\nRunning workload {num}: {workload.name}...")

        result = run_workload_benchmark(
            num,
            iterations=args.iterations,
            warmup=args.warmup,
            run_langgraph=run_lg,
            run_apxm=run_apxm,
        )
        results["workloads"][workload.name] = result

    # Output results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        if not args.json:
            print(f"\nResults saved to: {args.output}")

    if args.json:
        print(json.dumps(results, indent=2))
    elif not args.output:
        # Print summary
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        for name, result in results["workloads"].items():
            print(f"\n{name}:")
            if "error" in result:
                print(f"  Error: {result['error']}")
            else:
                lg_result = result.get("results", {}).get("langgraph", {})
                apxm_result = result.get("results", {}).get("apxm", {})

                if lg_result and "error" not in lg_result:
                    if "mean_ms" in lg_result:
                        print(f"  LangGraph: {lg_result['mean_ms']:.2f} ms")
                    elif "note" in lg_result:
                        print(f"  LangGraph: {lg_result['note']}")

                if apxm_result and "error" not in apxm_result:
                    if "mean_ms" in apxm_result:
                        print(f"  A-PXM: {apxm_result['mean_ms']:.2f} ms")
                    elif apxm_result.get("success"):
                        print(f"  A-PXM: {apxm_result.get('mean_ms', 'N/A')} ms")
                    elif "note" in apxm_result:
                        print(f"  A-PXM: {apxm_result['note']}")
                    elif "error" in apxm_result:
                        print(f"  A-PXM: Error - {apxm_result['error'][:50]}...")


if __name__ == "__main__":
    main()
