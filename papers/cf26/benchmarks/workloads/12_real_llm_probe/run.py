#!/usr/bin/env python3
"""
Real-LLM Probe Benchmark

Measures actual LLM latency, token usage, and scheduler overhead with real Ollama calls.
Generates data for tab/real-llm.tex.

Requirements:
- Ollama running with a configured model
- APXM config with Ollama backend

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
DEFAULT_ITERATIONS = 5
DEFAULT_WARMUP = 2


# Workflow source file
WORKFLOW_FILE = Path(__file__).parent / "workflow.ais"


def get_workflow_source() -> str:
    """Read workflow source from .ais file."""
    return WORKFLOW_FILE.read_text()


def find_apxm_cli() -> Path:
    """Find the apxm CLI binary using shared utility."""
    cli = _find_cli()
    if cli is None:
        raise FileNotFoundError("apxm CLI not found. Build with: python tools/apxm_cli.py compiler build")
    return cli


def check_ollama_running() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def run_llm_probe(cli_path: Path, opt_level: int = 1) -> dict:
    """Run the LLM probe workflow and capture metrics.

    Returns full metrics data including runtime-only time extracted from link_phases.
    """
    sample_timestamp = datetime.now(timezone.utc).isoformat()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ais_file = tmpdir / "probe.ais"
        metrics_file = tmpdir / "metrics.json"

        # Write probe workflow
        ais_file.write_text(get_workflow_source())

        # Set up environment using shared utilities
        config = ApxmConfig.detect()
        if config.conda_prefix:
            env = setup_mlir_environment(config.conda_prefix, config.target_dir)
        else:
            env = os.environ.copy()

        # Run with metrics
        cmd = [
            str(cli_path),
            "run",
            str(ais_file),
            f"-O{opt_level}",
            "--emit-metrics", str(metrics_file),
        ]

        start = time.perf_counter()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # LLM calls can be slow
            env=env,
        )
        wall_time_ms = (time.perf_counter() - start) * 1000

        if result.returncode != 0:
            return {
                "success": False,
                "timestamp": sample_timestamp,
                "error": result.stderr or result.stdout,
                "wall_time_ms": wall_time_ms,
            }

        # Parse stdout for execution info
        stdout = result.stdout
        nodes_executed = 0
        duration_ms = 0

        # Parse "Executed N nodes in M ms"
        match = re.search(r"Executed (\d+) nodes in (\d+) ms", stdout)
        if match:
            nodes_executed = int(match.group(1))
            duration_ms = int(match.group(2))

        # Parse LLM usage info (if metrics feature enabled)
        input_tokens = 0
        output_tokens = 0
        llm_requests = 0

        match = re.search(r"LLM usage: (\d+) input, (\d+) output", stdout)
        if match:
            input_tokens = int(match.group(1))
            output_tokens = int(match.group(2))

        match = re.search(r"LLM requests: (\d+)", stdout)
        if match:
            llm_requests = int(match.group(1))

        # Parse scheduler overhead
        scheduler_overhead_us = 0
        match = re.search(r"Scheduler overhead: ([\d.]+)μs/op", stdout)
        if match:
            scheduler_overhead_us = float(match.group(1))

        # Read metrics file if it exists (FULL metrics data)
        metrics_data = {}
        if metrics_file.exists():
            try:
                metrics_data = json.loads(metrics_file.read_text())
            except Exception:
                pass

        # Extract runtime_only_ms from link_phases if available
        runtime_only_ms = wall_time_ms
        link_phases = metrics_data.get("link_phases", {})
        if "runtime_ms" in link_phases:
            runtime_only_ms = link_phases["runtime_ms"]

        return {
            "success": True,
            "timestamp": sample_timestamp,
            "wall_time_ms": wall_time_ms,
            "runtime_only_ms": runtime_only_ms,
            "nodes_executed": nodes_executed,
            "duration_ms": duration_ms,
            "llm": {
                "requests": llm_requests,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            "scheduler_overhead_us": scheduler_overhead_us,
            "runtime": metrics_data,  # Full --emit-metrics output
            "output": {
                "stdout": stdout,
            },
        }


def run_langgraph(iterations: int = DEFAULT_ITERATIONS, warmup: int = DEFAULT_WARMUP) -> dict:
    """Entry point for suite runner - not applicable for LLM probe."""
    return {"note": "LLM probe only runs through A-PXM"}


def run_apxm(iterations: int = DEFAULT_ITERATIONS, warmup: int = DEFAULT_WARMUP) -> dict:
    """Entry point for the suite runner.

    Warmup iterations are executed but discarded from results.
    """
    try:
        cli_path = find_apxm_cli()
    except FileNotFoundError as e:
        return {"error": str(e)}

    if not check_ollama_running():
        return {"error": "Ollama not running. Start with: ollama serve"}

    samples = []
    total_iterations = warmup + iterations

    for i in range(total_iterations):
        is_warmup = i < warmup
        result = run_llm_probe(cli_path)
        result["iteration"] = i - warmup if not is_warmup else -(warmup - i)

        # Only store measurement iterations (discard warmup)
        if not is_warmup:
            samples.append(result)

    successful = [s for s in samples if s["success"]]
    if not successful:
        return {"error": "No successful runs", "samples": samples}

    # Compute summary from successful samples
    runtime_only_values = [s["runtime_only_ms"] for s in successful]
    wall_time_values = [s["wall_time_ms"] for s in successful]
    scheduler_overheads = [s["scheduler_overhead_us"] for s in successful]
    input_tokens = [s["llm"]["input_tokens"] for s in successful]
    output_tokens = [s["llm"]["output_tokens"] for s in successful]

    # Calculate overhead ratio
    overhead_ratio = 0
    if runtime_only_values and scheduler_overheads:
        avg_runtime_us = statistics.mean(runtime_only_values) * 1000
        avg_sched_us = statistics.mean(scheduler_overheads)
        overhead_ratio = (avg_sched_us / avg_runtime_us) * 100 if avg_runtime_us > 0 else 0

    return {
        "config": {
            "iterations": iterations,
            "warmup": warmup,
        },
        "samples": samples,
        "summary": {
            "successful_runs": len(successful),
            "failed_runs": len(samples) - len(successful),
            "runtime_only": {
                "mean_ms": statistics.mean(runtime_only_values),
                "std_ms": statistics.stdev(runtime_only_values) if len(runtime_only_values) > 1 else 0,
            },
            "wall_time": {
                "mean_ms": statistics.mean(wall_time_values),
                "std_ms": statistics.stdev(wall_time_values) if len(wall_time_values) > 1 else 0,
            },
            "scheduler_overhead": {
                "mean_us": statistics.mean(scheduler_overheads) if scheduler_overheads else 0,
            },
            "overhead_ratio_pct": overhead_ratio,
            "tokens": {
                "input_mean": statistics.mean(input_tokens) if input_tokens else 0,
                "output_mean": statistics.mean(output_tokens) if output_tokens else 0,
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Real-LLM Probe Benchmark")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                        help=f"Number of measurement iterations (default: {DEFAULT_ITERATIONS})")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help=f"Number of warmup iterations to discard (default: {DEFAULT_WARMUP})")
    args = parser.parse_args()

    try:
        cli_path = find_apxm_cli()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not check_ollama_running():
        print("Error: Ollama not running. Start with: ollama serve", file=sys.stderr)
        sys.exit(1)

    results = {
        "meta": {
            "benchmark": "real_llm_probe",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.0",
        },
        "config": {
            "iterations": args.iterations,
            "warmup": args.warmup,
        },
        "input": {
            "workflow_source": get_workflow_source().strip(),
        },
        "samples": [],
    }

    if not args.json:
        print(f"\nReal-LLM Probe Benchmark")
        print(f"{'=' * 60}")
        print(f"Measurement iterations: {args.iterations}")
        print(f"Warmup iterations: {args.warmup} (discarded)")
        print()

    total_iterations = args.warmup + args.iterations

    for i in range(total_iterations):
        is_warmup = i < args.warmup
        iteration_num = i - args.warmup if not is_warmup else -(args.warmup - i)

        if not args.json:
            phase = "warmup" if is_warmup else "run"
            run_num = i + 1 if is_warmup else (i - args.warmup + 1)
            total = args.warmup if is_warmup else args.iterations
            print(f"{phase.capitalize()} {run_num}/{total}...", end=" ", flush=True)

        result = run_llm_probe(cli_path)
        result["iteration"] = iteration_num

        if not args.json:
            if result["success"]:
                print(f"{result['runtime_only_ms']:.0f} ms (sched: {result['scheduler_overhead_us']:.2f} μs/op)")
            else:
                print(f"ERROR: {result.get('error', 'Unknown')[:50]}")

        # Only store measurement iterations (discard warmup)
        if not is_warmup:
            results["samples"].append(result)

    # Compute summary from successful samples
    successful = [s for s in results["samples"] if s["success"]]
    if successful:
        runtime_only_values = [s["runtime_only_ms"] for s in successful]
        wall_time_values = [s["wall_time_ms"] for s in successful]
        scheduler_overheads = [s["scheduler_overhead_us"] for s in successful]
        input_tokens = [s["llm"]["input_tokens"] for s in successful]
        output_tokens = [s["llm"]["output_tokens"] for s in successful]

        overhead_ratio = 0
        if runtime_only_values and scheduler_overheads:
            avg_runtime_us = statistics.mean(runtime_only_values) * 1000
            avg_sched_us = statistics.mean(scheduler_overheads)
            overhead_ratio = (avg_sched_us / avg_runtime_us) * 100 if avg_runtime_us > 0 else 0

        results["summary"] = {
            "successful_runs": len(successful),
            "failed_runs": len(results["samples"]) - len(successful),
            "runtime_only": {
                "mean_ms": statistics.mean(runtime_only_values),
                "std_ms": statistics.stdev(runtime_only_values) if len(runtime_only_values) > 1 else 0,
            },
            "wall_time": {
                "mean_ms": statistics.mean(wall_time_values),
                "std_ms": statistics.stdev(wall_time_values) if len(wall_time_values) > 1 else 0,
            },
            "scheduler_overhead": {
                "mean_us": statistics.mean(scheduler_overheads) if scheduler_overheads else 0,
            },
            "overhead_ratio_pct": overhead_ratio,
            "tokens": {
                "input_mean": statistics.mean(input_tokens) if input_tokens else 0,
                "output_mean": statistics.mean(output_tokens) if output_tokens else 0,
            },
        }

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print()
        print(f"{'=' * 60}")
        print(f"SUMMARY (for tab/real-llm.tex)")
        print(f"{'=' * 60}")

        if successful:
            summary = results["summary"]
            print(f"Runtime Only (mean):     {summary['runtime_only']['mean_ms']:.0f} ms")
            print(f"Runtime Only (std):      {summary['runtime_only']['std_ms']:.0f} ms")
            print(f"Scheduler Overhead:      {summary['scheduler_overhead']['mean_us']:.2f} μs/op")
            print(f"Overhead Ratio:          {summary['overhead_ratio_pct']:.4f}%")
            print(f"Input Tokens (mean):     {summary['tokens']['input_mean']:.0f}")
            print(f"Output Tokens (mean):    {summary['tokens']['output_mean']:.0f}")
        else:
            print("No successful runs")


if __name__ == "__main__":
    main()
