#!/usr/bin/env python3
"""
FuseReasoning Quality Benchmark

Compares O0 (no fusion) vs O1 (with FuseReasoning) for different task types.
Generates data for tab/fusion-applicability.tex.

Task types tested:
- Classification: Multiple parallel classification queries
- Extraction: Multiple parallel extraction queries
- Reasoning: Sequential reasoning chain
- Creative: Creative writing tasks

Run: python run.py [--json] [--iterations N]
"""

import argparse
import json
import os
import re
import subprocess
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to import utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from apxm_runner import find_apxm_cli as _find_cli

# Add tools directory for shared utilities
_tools_dir = Path(__file__).parent.parent.parent.parent.parent.parent / "tools"
sys.path.insert(0, str(_tools_dir))
from apxm_env import ApxmConfig, setup_mlir_environment

# Default configuration
DEFAULT_ITERATIONS = 3


# Task type workflows - now read from .ais files
WORKFLOW_DIR = Path(__file__).parent

# Task type configurations (workflow content is loaded from .ais files)
TASK_TYPES = {
    "classification": {
        "description": "Parallel classification queries",
        "file": "classification.ais",
    },
    "extraction": {
        "description": "Parallel entity extraction",
        "file": "extraction.ais",
    },
    "reasoning": {
        "description": "Sequential reasoning chain (with dependencies)",
        "file": "reasoning.ais",
    },
    "creative": {
        "description": "Creative generation",
        "file": "creative.ais",
    },
}


def get_workflow(task_name: str) -> str:
    """Read workflow source from .ais file."""
    task_config = TASK_TYPES[task_name]
    workflow_file = WORKFLOW_DIR / task_config["file"]
    return workflow_file.read_text()


def find_apxm_cli() -> Path:
    """Find the apxm CLI binary using shared utility."""
    cli = _find_cli()
    if cli is None:
        raise FileNotFoundError("apxm CLI not found. Build with: python tools/apxm_cli.py compiler build")
    return cli


def run_workflow(cli_path: Path, workflow_content: str, opt_level: int) -> dict:
    """Run a workflow and return timing info."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ais_file = tmpdir / "test.ais"
        ais_file.write_text(workflow_content)

        # Set up environment using shared utilities
        config = ApxmConfig.detect()
        if config.conda_prefix:
            env = setup_mlir_environment(config.conda_prefix, config.target_dir)
        else:
            env = os.environ.copy()

        cmd = [str(cli_path), "run", str(ais_file), f"-O{opt_level}"]

        start = time.perf_counter()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        wall_time_ms = (time.perf_counter() - start) * 1000

        if result.returncode != 0:
            return {"success": False, "error": result.stderr, "wall_time_ms": wall_time_ms}

        # Parse execution time from output
        duration_ms = wall_time_ms
        match = re.search(r"Executed \d+ nodes in (\d+) ms", result.stdout)
        if match:
            duration_ms = int(match.group(1))

        return {
            "success": True,
            "wall_time_ms": wall_time_ms,
            "duration_ms": duration_ms,
            "opt_level": opt_level,
        }


def benchmark_task_type(cli_path: Path, task_name: str, task_config: dict, iterations: int) -> dict:
    """Benchmark a task type with O0 and O1."""
    workflow = get_workflow(task_name)

    o0_samples = []
    o1_samples = []

    for _ in range(iterations):
        # Run with O0 (no fusion)
        o0_result = run_workflow(cli_path, workflow, opt_level=0)
        if o0_result["success"]:
            o0_samples.append(o0_result["duration_ms"])

        # Run with O1 (with fusion)
        o1_result = run_workflow(cli_path, workflow, opt_level=1)
        if o1_result["success"]:
            o1_samples.append(o1_result["duration_ms"])

    result = {
        "task_type": task_name,
        "description": task_config["description"],
        "iterations": iterations,
    }

    if o0_samples and o1_samples:
        o0_mean = statistics.mean(o0_samples)
        o1_mean = statistics.mean(o1_samples)
        speedup = o0_mean / o1_mean if o1_mean > 0 else 0

        result["o0_unfused"] = {
            "mean_ms": o0_mean,
            "std_ms": statistics.stdev(o0_samples) if len(o0_samples) > 1 else 0,
        }
        result["o1_fused"] = {
            "mean_ms": o1_mean,
            "std_ms": statistics.stdev(o1_samples) if len(o1_samples) > 1 else 0,
        }
        result["speedup"] = speedup
        result["recommended"] = speedup > 1.1  # Recommend if >10% speedup
    else:
        result["error"] = "Insufficient successful runs"

    return result


def run_langgraph(iterations: int = DEFAULT_ITERATIONS) -> dict:
    """Entry point for suite runner - not applicable for fusion quality."""
    return {"note": "Fusion quality only applies to A-PXM"}


def run_apxm(iterations: int = DEFAULT_ITERATIONS) -> dict:
    """Entry point for the suite runner."""
    try:
        cli_path = find_apxm_cli()
    except FileNotFoundError as e:
        return {"error": str(e)}

    results = []
    for task_name, task_config in TASK_TYPES.items():
        result = benchmark_task_type(cli_path, task_name, task_config, iterations)
        results.append(result)

    return {
        "task_types": list(TASK_TYPES.keys()),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="FuseReasoning Quality Benchmark")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                        help=f"Number of iterations (default: {DEFAULT_ITERATIONS})")
    parser.add_argument("--task", type=str, choices=list(TASK_TYPES.keys()),
                        help="Run only a specific task type")
    args = parser.parse_args()

    try:
        cli_path = find_apxm_cli()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    results = {
        "meta": {
            "benchmark": "fusion_quality",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "iterations": args.iterations,
        },
        "data": [],
    }

    task_types = {args.task: TASK_TYPES[args.task]} if args.task else TASK_TYPES

    if not args.json:
        print(f"\nFuseReasoning Quality Benchmark")
        print(f"{'=' * 70}")
        print(f"Comparing O0 (unfused) vs O1 (FuseReasoning enabled)")
        print(f"Iterations per task: {args.iterations}")
        print()

    for task_name, task_config in task_types.items():
        if not args.json:
            print(f"Testing {task_name}: {task_config['description']}...", flush=True)

        result = benchmark_task_type(cli_path, task_name, task_config, args.iterations)
        results["data"].append(result)

        if not args.json:
            if "error" in result:
                print(f"  ERROR: {result['error']}")
            else:
                print(f"  O0: {result['o0_unfused']['mean_ms']:.0f}ms, O1: {result['o1_fused']['mean_ms']:.0f}ms, Speedup: {result['speedup']:.2f}x")

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print()
        print(f"{'=' * 70}")
        print(f"SUMMARY (for tab/fusion-applicability.tex)")
        print(f"{'=' * 70}")
        print()
        print(f"{'Task Type':<15} | {'O0 (ms)':>10} | {'O1 (ms)':>10} | {'Speedup':>8} | {'Recommend':>10}")
        print(f"{'-'*15}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}")

        for r in results["data"]:
            if "error" in r:
                print(f"{r['task_type']:<15} | {'ERROR':>10} | {'':>10} | {'':>8} | {'':>10}")
            else:
                rec = "Yes" if r.get("recommended") else "No"
                print(f"{r['task_type']:<15} | {r['o0_unfused']['mean_ms']:>10.0f} | {r['o1_fused']['mean_ms']:>10.0f} | {r['speedup']:>7.2f}x | {rec:>10}")


if __name__ == "__main__":
    main()
