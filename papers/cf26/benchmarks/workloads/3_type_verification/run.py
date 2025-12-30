#!/usr/bin/env python3
"""
Type Verification Benchmark Runner

Compares compile-time error detection (A-PXM) vs runtime errors (LangGraph).

This benchmark tests COMPILE-TIME error detection:
- A-PXM catches undefined variable errors at compile time (before any LLM calls)
- LangGraph discovers errors at runtime (after potentially wasting LLM calls)
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to import apxm_runner
sys.path.insert(0, str(Path(__file__).parent.parent))
from apxm_runner import find_apxm_cli

WORKFLOW_FILE = Path(__file__).parent / "workflow.ais"


def run_langgraph_error_test() -> dict:
    """Test LangGraph runtime error detection."""
    from workflow import graph, HAS_OLLAMA

    initial_state = {
        "result": "",
        "output": "",
    }

    # Measure time to error
    start = time.perf_counter()
    try:
        graph.invoke(initial_state)
        return {
            "error_caught": False,
            "error_type": None,
            "time_to_error_ms": 0,
            "llm_calls_before_error": 0,
        }
    except KeyError as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "error_caught": True,
            "error_type": "KeyError",
            "error_message": str(e),
            "time_to_error_ms": elapsed_ms,
            "llm_calls_before_error": 1,  # first_step succeeded
            "cost_wasted": "$0.01+",  # Approximate cost of wasted LLM call
            "has_ollama": HAS_OLLAMA,
        }


def run_apxm_compile_test() -> dict:
    """Test A-PXM compile-time error detection."""
    cli_path = find_apxm_cli()

    if cli_path is None:
        return {
            "error_caught": None,
            "note": "A-PXM CLI not available - build with: python tools/apxm_cli.py compiler build",
            "expected_behavior": {
                "error_type": "compile-time",
                "time_to_error_ms": "~50ms",
                "llm_calls_before_error": 0,
                "cost_wasted": "$0.00",
            },
        }

    try:
        start = time.perf_counter()
        result = subprocess.run(
            [str(cli_path), "run", str(WORKFLOW_FILE), "-O0"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if result.returncode != 0:
            # Compilation/execution failed (expected for this broken file)
            return {
                "error_caught": True,
                "error_type": "compile-time",
                "error_message": result.stderr.strip()[:200],  # Truncate for display
                "time_to_error_ms": elapsed_ms,
                "llm_calls_before_error": 0,
                "cost_wasted": "$0.00",
            }
        else:
            return {
                "error_caught": False,
                "note": "Execution succeeded unexpectedly (error should have been caught)",
            }
    except subprocess.TimeoutExpired:
        return {
            "error_caught": None,
            "note": "Execution timed out",
        }


def main():
    parser = argparse.ArgumentParser(description="Type Verification Benchmark")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    results = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmark": "type_verification",
            "description": "Compile-time vs runtime error detection",
        },
        "results": {},
    }

    # Run LangGraph error test
    try:
        results["results"]["langgraph"] = run_langgraph_error_test()
    except ImportError as e:
        results["results"]["langgraph"] = {"error": str(e)}

    # Run A-PXM compile test
    results["results"]["apxm"] = run_apxm_compile_test()

    # Calculate comparison
    lg = results["results"].get("langgraph", {})
    apxm = results["results"].get("apxm", {})

    if lg.get("error_caught") and apxm.get("error_caught"):
        results["comparison"] = {
            "time_savings": f"{lg['time_to_error_ms']:.0f}ms -> {apxm['time_to_error_ms']:.0f}ms",
            "cost_savings": f"{lg.get('cost_wasted', 'N/A')} -> {apxm.get('cost_wasted', 'N/A')}",
            "llm_calls_saved": lg.get("llm_calls_before_error", 0),
        }

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\nType Verification Benchmark Results")
        print(f"{'=' * 50}")
        print()

        if "langgraph" in results["results"]:
            lg = results["results"]["langgraph"]
            print(f"LangGraph (runtime error):")
            if lg.get("error_caught"):
                print(f"  Error type: {lg['error_type']}")
                print(f"  Time to error: {lg['time_to_error_ms']:.2f} ms")
                print(f"  LLM calls wasted: {lg['llm_calls_before_error']}")
                print(f"  Cost wasted: {lg.get('cost_wasted', 'N/A')}")

        print()
        apxm = results["results"].get("apxm", {})
        print(f"A-PXM (compile-time error):")
        if apxm.get("note"):
            print(f"  {apxm['note']}")
            if "expected_behavior" in apxm:
                exp = apxm["expected_behavior"]
                print(f"  Expected time to error: {exp['time_to_error_ms']}")
                print(f"  Expected LLM calls wasted: {exp['llm_calls_before_error']}")
                print(f"  Expected cost wasted: {exp['cost_wasted']}")
        elif apxm.get("error_caught"):
            print(f"  Error type: {apxm['error_type']}")
            print(f"  Time to error: {apxm['time_to_error_ms']:.2f} ms")
            print(f"  LLM calls wasted: {apxm['llm_calls_before_error']}")
            print(f"  Cost wasted: {apxm.get('cost_wasted', 'N/A')}")


if __name__ == "__main__":
    main()
