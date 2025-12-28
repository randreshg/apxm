#!/usr/bin/env python3
"""
Master Benchmark Runner

Runs all DSL comparison benchmarks and outputs unified JSON results.

Usage:
    python runner.py --json                    # Run all, output JSON
    python runner.py --workload 1              # Run specific workload
    python runner.py --list                    # List available workloads
    python runner.py --output results.json     # Save to file
"""

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path


WORKLOADS = {
    1: {
        "name": "parallel_research",
        "dir": "1_parallel_research",
        "description": "Automatic parallelism from dataflow",
    },
    2: {
        "name": "chain_fusion",
        "dir": "2_chain_fusion",
        "description": "FuseReasoning compiler optimization (5 calls -> 1)",
    },
    3: {
        "name": "type_verification",
        "dir": "3_type_verification",
        "description": "Compile-time vs runtime error detection",
    },
    4: {
        "name": "scalability",
        "dir": "4_scalability",
        "description": "N-way parallelism efficiency (N=2,4,8)",
    },
    5: {
        "name": "memory_augmented",
        "dir": "5_memory_augmented",
        "description": "3-tier memory system (STM/LTM/Episodic)",
    },
    6: {
        "name": "tool_invocation",
        "dir": "6_tool_invocation",
        "description": "Native INV operations with capability system",
    },
    7: {
        "name": "reflection",
        "dir": "7_reflection",
        "description": "Built-in reflect operation for self-improvement",
    },
    8: {
        "name": "planning",
        "dir": "8_planning",
        "description": "Native plan operation for task decomposition",
    },
    9: {
        "name": "conditional_routing",
        "dir": "9_conditional_routing",
        "description": "Dataflow-based parallel preparation and routing",
    },
    10: {
        "name": "multi_agent",
        "dir": "10_multi_agent",
        "description": "Native agent definitions with communicate operations",
    },
}


def get_platform_info() -> dict:
    """Collect platform information."""
    import platform

    return {
        "os": platform.system(),
        "os_version": platform.release(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
    }


def run_workload(workload_num: int, iterations: int = 10) -> dict:
    """Run a specific workload and return results."""
    if workload_num not in WORKLOADS:
        return {"error": f"Unknown workload: {workload_num}"}

    workload = WORKLOADS[workload_num]
    workload_dir = Path(__file__).parent / workload["dir"]
    run_script = workload_dir / "run.py"

    if not run_script.exists():
        return {"error": f"Run script not found: {run_script}"}

    # Import and run the workload
    spec = importlib.util.spec_from_file_location("run", run_script)
    module = importlib.util.module_from_spec(spec)

    # Add workload dir to path for imports
    sys.path.insert(0, str(workload_dir))

    try:
        spec.loader.exec_module(module)

        # Build results by running workload components
        results = {
            "workload": workload["name"],
            "description": workload["description"],
        }

        # Try to run LangGraph benchmark
        if hasattr(module, "run_langgraph"):
            try:
                results["langgraph"] = module.run_langgraph(iterations)
            except Exception as e:
                results["langgraph"] = {"error": str(e)}

        # Try to run A-PXM benchmark
        if hasattr(module, "run_apxm"):
            try:
                results["apxm"] = module.run_apxm(iterations)
            except Exception as e:
                results["apxm"] = {"error": str(e)}

        return results

    except Exception as e:
        return {"error": str(e)}

    finally:
        sys.path.pop(0)


def main():
    parser = argparse.ArgumentParser(
        description="A-PXM vs LangGraph DSL Comparison Benchmark Runner"
    )
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--list", action="store_true", help="List available workloads")
    parser.add_argument("--workload", type=int, help="Run specific workload (1-10)")
    parser.add_argument("--iterations", type=int, default=10, help="Iterations per benchmark")
    parser.add_argument("--output", type=str, help="Output file path")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable Workloads:")
        print("=" * 60)
        for num, info in WORKLOADS.items():
            print(f"  {num}. {info['name']}")
            print(f"     {info['description']}")
            print()
        return

    # Build full results
    results = {
        "meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "benchmark_suite": "apxm_dsl_comparison",
            "platform": get_platform_info(),
        },
        "config": {
            "iterations": args.iterations,
        },
        "workloads": {},
    }

    # Run specified workload or all
    workloads_to_run = [args.workload] if args.workload else list(WORKLOADS.keys())

    for num in workloads_to_run:
        if not args.json:
            print(f"\nRunning workload {num}: {WORKLOADS[num]['name']}...")

        result = run_workload(num, args.iterations)
        results["workloads"][WORKLOADS[num]["name"]] = result

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
                if "langgraph" in result and "error" not in result.get("langgraph", {}):
                    lg = result["langgraph"]
                    if "mean_ms" in lg:
                        print(f"  LangGraph: {lg['mean_ms']:.2f} ms")
                if "apxm" in result:
                    apxm = result["apxm"]
                    if "note" in apxm:
                        print(f"  A-PXM: {apxm['note']}")


if __name__ == "__main__":
    main()
