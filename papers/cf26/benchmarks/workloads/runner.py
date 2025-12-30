#!/usr/bin/env python3
"""
Master Benchmark Runner

Runs all DSL comparison benchmarks and outputs unified JSON results.
Applies consistent JSON schema v2.0 to ALL workloads:
- Warmup iterations are executed but discarded from results
- Per-iteration full data is stored in samples array
- Summary statistics computed from measurement samples only

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
import inspect
import statistics
from datetime import datetime, timezone
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
    11: {
        "name": "compilation_scaling",
        "dir": "11_compilation_scaling",
        "description": "Compilation phase timing at different op counts",
    },
    12: {
        "name": "real_llm_probe",
        "dir": "12_real_llm_probe",
        "description": "Real LLM latency and token measurement",
    },
    13: {
        "name": "fusion_quality",
        "dir": "13_fusion_quality",
        "description": "FuseReasoning optimization effectiveness",
    },
    14: {
        "name": "token_estimation",
        "dir": "14_token_estimation",
        "description": "Token cost estimation before/after fusion",
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


def compute_summary(samples: list[dict]) -> dict:
    """Compute summary statistics from successful samples."""
    successful = [s for s in samples if s.get("success", True) and "error" not in s]
    if not successful:
        return {"successful_runs": 0, "failed_runs": len(samples)}

    # Extract timing values (try multiple field names)
    times = []
    for s in successful:
        for key in ["mean_ms", "wall_time_ms", "duration_ms", "runtime_only_ms"]:
            if key in s:
                times.append(s[key])
                break

    summary = {
        "successful_runs": len(successful),
        "failed_runs": len(samples) - len(successful),
    }

    if times:
        summary["timing"] = {
            "mean_ms": statistics.mean(times),
            "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
            "min_ms": min(times),
            "max_ms": max(times),
        }

    return summary


def run_workload(workload_num: int, iterations: int = 10, warmup: int = 3) -> dict:
    """Run a specific workload and return results with unified schema v2.0.

    Schema v2.0:
    - Warmup iterations executed but discarded
    - Per-iteration data stored in samples array
    - Summary computed from measurement samples only
    """
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

        # Build results with unified schema
        results = {
            "workload": workload["name"],
            "description": workload["description"],
            "config": {
                "iterations": iterations,
                "warmup": warmup,
            },
        }

        # Try to run LangGraph benchmark
        if hasattr(module, "run_langgraph"):
            try:
                fn = module.run_langgraph
                sig = inspect.signature(fn)
                if "warmup" in sig.parameters:
                    raw_result = fn(iterations, warmup=warmup)
                else:
                    raw_result = fn(iterations)

                # Wrap result in unified schema if not already
                if isinstance(raw_result, dict):
                    if "samples" in raw_result:
                        results["langgraph"] = raw_result
                    elif "error" in raw_result or "note" in raw_result:
                        results["langgraph"] = raw_result
                    else:
                        # Legacy format - wrap it
                        results["langgraph"] = {
                            "data": raw_result,
                            "summary": compute_summary([raw_result]) if raw_result else {},
                        }
                else:
                    results["langgraph"] = {"data": raw_result}
            except Exception as e:
                results["langgraph"] = {"error": str(e)}

        # Try to run A-PXM benchmark
        if hasattr(module, "run_apxm"):
            try:
                fn = module.run_apxm
                sig = inspect.signature(fn)
                if "warmup" in sig.parameters:
                    raw_result = fn(iterations, warmup=warmup)
                else:
                    raw_result = fn(iterations)

                # Wrap result in unified schema if not already
                if isinstance(raw_result, dict):
                    if "samples" in raw_result:
                        results["apxm"] = raw_result
                    elif "error" in raw_result or "note" in raw_result:
                        results["apxm"] = raw_result
                    else:
                        # Legacy format - wrap it
                        results["apxm"] = {
                            "data": raw_result,
                            "summary": compute_summary([raw_result]) if raw_result else {},
                        }
                else:
                    results["apxm"] = {"data": raw_result}
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
    parser.add_argument("--workload", type=int, help="Run specific workload (1-14)")
    parser.add_argument("--iterations", type=int, default=10, help="Iterations per benchmark")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations (not counted)")
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

    # Build full results with unified schema v2.0
    results = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmark_suite": "apxm_dsl_comparison",
            "version": "2.0",
            "platform": get_platform_info(),
        },
        "config": {
            "iterations": args.iterations,
            "warmup": args.warmup,
            "note": "Warmup iterations are executed but discarded from results",
        },
        "workloads": {},
    }

    # Run specified workload or all
    workloads_to_run = [args.workload] if args.workload else list(WORKLOADS.keys())

    for num in workloads_to_run:
        if not args.json:
            print(f"\nRunning workload {num}: {WORKLOADS[num]['name']}...")

        result = run_workload(num, args.iterations, warmup=args.warmup)
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
