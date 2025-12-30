#!/usr/bin/env python3
"""
Compilation Scaling Benchmark

Generates data for tab/compilation-scaling.tex by measuring compilation time
at different operation counts (10, 25, 50, 100 operations).

Uses the `apxm compile --emit-diagnostics` flag to get phase timings.

Run: python run.py [--json] [--iterations N]
"""

import argparse
import json
import os
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
DEFAULT_ITERATIONS = 10
DEFAULT_WARMUP = 2
OP_COUNTS = [10, 25, 50, 100]


def find_apxm_cli() -> Path:
    """Find the apxm CLI binary using shared utility."""
    cli = _find_cli()
    if cli is None:
        raise FileNotFoundError("apxm CLI not found. Build with: python tools/apxm_cli.py compiler build")
    return cli


def generate_synthetic_ais(num_ops: int) -> str:
    """Generate a synthetic AIS file with N RSN operations."""
    lines = [
        f"// Synthetic AIS file with {num_ops} operations",
        f"// Generated for compilation scaling benchmark",
        "",
        "agent ScalingTest {",
        "    flow main {",
    ]

    # Generate parallel RSN operations
    result_names = []
    for i in range(num_ops):
        result_name = f"result_{i}"
        result_names.append(result_name)
        lines.append(f'        rsn "Task {i}: Process data element" -> {result_name}')

    # Add merge to consume all outputs (prevents dead code)
    lines.append("")
    lines.append(f"        // Merge all parallel outputs")
    lines.append(f"        merge [{', '.join(result_names)}] -> final")

    lines.extend([
        "    }",
        "}",
    ])

    return "\n".join(lines)


def run_compilation(cli_path: Path, ais_content: str, opt_level: int = 1) -> dict:
    """Run compilation and return diagnostics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ais_file = tmpdir / "test.ais"
        diag_file = tmpdir / "diagnostics.json"
        output_file = tmpdir / "output.apxmobj"

        # Write AIS content
        ais_file.write_text(ais_content)

        # Set up environment using shared utilities
        config = ApxmConfig.detect()
        if config.conda_prefix:
            env = setup_mlir_environment(config.conda_prefix, config.target_dir)
        else:
            env = os.environ.copy()

        # Run compilation
        cmd = [
            str(cli_path),
            "compile",
            str(ais_file),
            f"-O{opt_level}",
            "--emit-diagnostics", str(diag_file),
            "-o", str(output_file),
        ]

        # Debug: print command
        # print(f"DEBUG: Running: {' '.join(cmd)}", file=sys.stderr)

        start = time.perf_counter()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        wall_time_ms = (time.perf_counter() - start) * 1000

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout
            return {
                "success": False,
                "error": error_msg,
                "wall_time_ms": wall_time_ms,
            }

        # Read diagnostics
        try:
            diagnostics = json.loads(diag_file.read_text())
            return {
                "success": True,
                "wall_time_ms": wall_time_ms,
                "diagnostics": diagnostics,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "wall_time_ms": wall_time_ms,
            }


def benchmark_op_count(cli_path: Path, num_ops: int, iterations: int, warmup: int = DEFAULT_WARMUP, opt_level: int = 1) -> dict:
    """Benchmark compilation for a specific operation count.

    Warmup iterations are executed but discarded from results.
    Per-iteration full data is stored in samples array.
    """
    ais_content = generate_synthetic_ais(num_ops)

    samples = []
    total_iterations = warmup + iterations

    for i in range(total_iterations):
        is_warmup = i < warmup
        sample_timestamp = datetime.now(timezone.utc).isoformat()

        result = run_compilation(cli_path, ais_content, opt_level)

        # Build sample with full data
        sample = {
            "iteration": i - warmup if not is_warmup else -(warmup - i),  # Negative for warmup
            "timestamp": sample_timestamp,
            "success": result["success"],
            "wall_time_ms": result["wall_time_ms"],
        }

        if result["success"]:
            sample["compiler"] = {
                "diagnostics": result["diagnostics"]  # Full diagnostics
            }
        else:
            sample["error"] = result.get("error", "Unknown error")

        # Only store measurement iterations (discard warmup)
        if not is_warmup:
            samples.append(sample)

    # Compute summary from successful samples
    successful = [s for s in samples if s["success"]]
    if not successful:
        return {
            "num_ops": num_ops,
            "error": "No successful runs",
            "input": {"workflow_source": ais_content, "opt_level": opt_level},
            "config": {"iterations": iterations, "warmup": warmup},
            "samples": samples,
        }

    wall_times = [s["wall_time_ms"] for s in successful]
    total_ms_values = [
        s["compiler"]["diagnostics"].get("compilation_phases", {}).get("total_ms", 0)
        for s in successful
    ]
    artifact_gen_values = [
        s["compiler"]["diagnostics"].get("compilation_phases", {}).get("artifact_gen_ms", 0)
        for s in successful
    ]
    parse_opt_values = [t - a for t, a in zip(total_ms_values, artifact_gen_values)]

    return {
        "num_ops": num_ops,
        "input": {
            "workflow_source": ais_content,
            "opt_level": opt_level,
        },
        "config": {
            "iterations": iterations,
            "warmup": warmup,
        },
        "samples": samples,
        "summary": {
            "successful_runs": len(successful),
            "failed_runs": len(samples) - len(successful),
            "wall_time": {
                "mean_ms": statistics.mean(wall_times),
                "std_ms": statistics.stdev(wall_times) if len(wall_times) > 1 else 0,
                "min_ms": min(wall_times),
                "max_ms": max(wall_times),
            },
            "compile_phases": {
                "parse_opt_mean_ms": statistics.mean(parse_opt_values) if parse_opt_values else 0,
                "artifact_gen_mean_ms": statistics.mean(artifact_gen_values) if artifact_gen_values else 0,
                "total_mean_ms": statistics.mean(total_ms_values) if total_ms_values else 0,
            },
        },
    }


def run_langgraph(iterations: int = DEFAULT_ITERATIONS, warmup: int = DEFAULT_WARMUP) -> dict:
    """Entry point for suite runner - returns empty dict as LangGraph doesn't have compilation."""
    return {"note": "LangGraph has no compilation phase"}


def run_apxm(iterations: int = DEFAULT_ITERATIONS, warmup: int = DEFAULT_WARMUP) -> dict:
    """Entry point for the suite runner."""
    try:
        cli_path = find_apxm_cli()
    except FileNotFoundError as e:
        return {"error": str(e)}

    results = []
    for num_ops in OP_COUNTS:
        result = benchmark_op_count(cli_path, num_ops, iterations, warmup=warmup)
        results.append(result)

    return {
        "op_counts": OP_COUNTS,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Compilation Scaling Benchmark")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                        help=f"Number of measurement iterations (default: {DEFAULT_ITERATIONS})")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help=f"Number of warmup iterations to discard (default: {DEFAULT_WARMUP})")
    parser.add_argument("--opt-level", type=int, default=1,
                        help="Optimization level (default: 1)")
    args = parser.parse_args()

    try:
        cli_path = find_apxm_cli()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    results = {
        "meta": {
            "benchmark": "compilation_scaling",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.0",
        },
        "config": {
            "iterations": args.iterations,
            "warmup": args.warmup,
            "opt_level": args.opt_level,
            "op_counts": OP_COUNTS,
        },
        "data": [],
    }

    if not args.json:
        print(f"\nCompilation Scaling Benchmark")
        print(f"{'=' * 60}")
        print(f"Measurement iterations: {args.iterations}")
        print(f"Warmup iterations: {args.warmup} (discarded)")
        print(f"Optimization level: O{args.opt_level}")
        print()

    for num_ops in OP_COUNTS:
        if not args.json:
            print(f"Benchmarking {num_ops} operations...", end=" ", flush=True)

        result = benchmark_op_count(cli_path, num_ops, args.iterations, args.warmup, args.opt_level)
        results["data"].append(result)

        if not args.json:
            if "error" in result and "summary" not in result:
                print(f"ERROR: {result['error']}")
            else:
                phases = result["summary"]["compile_phases"]
                print(f"{phases['total_mean_ms']:.2f} ms")

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print()
        print(f"{'Ops':>5} | {'Parse+Opt (ms)':>14} | {'Artifact (ms)':>13} | {'Total (ms)':>12}")
        print(f"{'-'*5}-+-{'-'*14}-+-{'-'*13}-+-{'-'*12}")

        for r in results["data"]:
            if "error" in r and "summary" not in r:
                print(f"{r['num_ops']:>5} | {'ERROR':>14} | {'':>13} | {'':>12}")
            else:
                phases = r["summary"]["compile_phases"]
                print(f"{r['num_ops']:>5} | {phases['parse_opt_mean_ms']:>14.2f} | {phases['artifact_gen_mean_ms']:>13.2f} | {phases['total_mean_ms']:>12.2f}")

        print()
        print("Data ready for tab/compilation-scaling.tex")


if __name__ == "__main__":
    main()
