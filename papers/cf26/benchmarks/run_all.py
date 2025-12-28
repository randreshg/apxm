#!/usr/bin/env python3
"""
A-PXM Benchmark Suite Runner

Unified script to run all benchmarks and generate analysis.

Usage:
    python run_all.py                    # Run everything
    python run_all.py --workloads        # Run only DSL comparison workloads
    python run_all.py --runtime          # Run only Rust runtime benchmarks
    python run_all.py --analyze          # Run only analysis on existing results
    python run_all.py --quick            # Quick mode (fewer iterations)
    python run_all.py --workload 1,2,5   # Run specific workloads
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Paths
BENCHMARKS_DIR = Path(__file__).parent
WORKLOADS_DIR = BENCHMARKS_DIR / "workloads"
RUNTIME_DIR = BENCHMARKS_DIR / "runtime"
RESULTS_DIR = BENCHMARKS_DIR / "results"
ANALYSIS_DIR = BENCHMARKS_DIR / "analysis"
CONFIG_PATH = BENCHMARKS_DIR / "config.json"


def load_config() -> dict:
    """Load benchmark configuration."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {
        "defaults": {"iterations": 10, "warmup": 3},
        "llm": {"provider": "ollama", "model": "phi3:mini"},
    }


def check_dependencies() -> dict:
    """Check availability of required dependencies."""
    deps = {
        "python": True,  # Obviously available
        "ollama": False,
        "apxm": False,
        "cargo": False,
        "langgraph": False,
    }

    # Check Ollama
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, timeout=5
        )
        deps["ollama"] = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check apxm CLI
    try:
        result = subprocess.run(
            ["apxm", "--version"], capture_output=True, timeout=5
        )
        deps["apxm"] = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check Cargo (for Rust runtime benchmarks)
    try:
        result = subprocess.run(
            ["cargo", "--version"], capture_output=True, timeout=5
        )
        deps["cargo"] = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check LangGraph
    try:
        import langgraph
        deps["langgraph"] = True
    except ImportError:
        pass

    return deps


def run_workloads(
    workload_nums: Optional[list] = None,
    iterations: int = 10,
    verbose: bool = True,
) -> dict:
    """Run DSL comparison workloads."""
    runner_script = WORKLOADS_DIR / "runner.py"

    if not runner_script.exists():
        return {"error": f"Runner script not found: {runner_script}"}

    cmd = [sys.executable, str(runner_script), "--json"]

    if workload_nums:
        # Run each specified workload
        all_results = {"workloads": {}}
        for num in workload_nums:
            if verbose:
                print(f"  Running workload {num}...")
            result = subprocess.run(
                cmd + ["--workload", str(num), "--iterations", str(iterations)],
                capture_output=True,
                text=True,
                cwd=WORKLOADS_DIR,
            )
            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    all_results["workloads"].update(data.get("workloads", {}))
                    all_results["meta"] = data.get("meta", {})
                    all_results["config"] = data.get("config", {})
                except json.JSONDecodeError:
                    all_results["workloads"][f"workload_{num}"] = {
                        "error": "Invalid JSON output"
                    }
            else:
                all_results["workloads"][f"workload_{num}"] = {
                    "error": result.stderr or "Unknown error"
                }
        return all_results
    else:
        # Run all workloads
        if verbose:
            print("  Running all workloads...")
        result = subprocess.run(
            cmd + ["--iterations", str(iterations)],
            capture_output=True,
            text=True,
            cwd=WORKLOADS_DIR,
        )
        if result.returncode == 0:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON output from runner"}
        return {"error": result.stderr or "Unknown error"}


def run_runtime_benchmarks(quick: bool = False, verbose: bool = True) -> dict:
    """Run Rust runtime benchmarks."""
    results = {"rust_benchmarks": {}}

    # Find the runtime crate
    crate_path = BENCHMARKS_DIR.parent.parent.parent / "crates" / "runtime" / "apxm-runtime"

    if not crate_path.exists():
        return {"error": f"Runtime crate not found: {crate_path}"}

    # Run cargo bench or examples
    examples = ["benchmark_overhead", "scalability_demo", "parallel_research_demo"]

    for example in examples:
        if verbose:
            print(f"  Running {example}...")

        # Try to run with --release
        cmd = [
            "cargo", "run", "--release",
            "--example", example,
            "-p", "apxm-runtime",
        ]

        if quick:
            # Set environment variable for quick mode
            env = os.environ.copy()
            env["BENCHMARK_QUICK"] = "1"
        else:
            env = None

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=crate_path.parent.parent.parent,  # Root of workspace
                timeout=300,  # 5 minute timeout
                env=env,
            )

            if result.returncode == 0:
                # Try to parse JSON from output
                output = result.stdout
                # Look for JSON in output (might have other text)
                try:
                    # Try to find JSON object in output
                    start = output.find("{")
                    end = output.rfind("}") + 1
                    if start >= 0 and end > start:
                        json_str = output[start:end]
                        results["rust_benchmarks"][example] = json.loads(json_str)
                    else:
                        results["rust_benchmarks"][example] = {
                            "raw_output": output[:2000],
                            "note": "No JSON found in output",
                        }
                except json.JSONDecodeError:
                    results["rust_benchmarks"][example] = {
                        "raw_output": output[:2000],
                        "note": "Could not parse JSON",
                    }
            else:
                results["rust_benchmarks"][example] = {
                    "error": result.stderr[:500] if result.stderr else "Unknown error",
                }
        except subprocess.TimeoutExpired:
            results["rust_benchmarks"][example] = {"error": "Timeout (5 min)"}
        except FileNotFoundError:
            results["rust_benchmarks"][example] = {"error": "Cargo not found"}

    return results


def run_analysis(results_file: Optional[Path] = None, verbose: bool = True) -> dict:
    """Run analysis scripts on benchmark results."""
    analysis_results = {}

    # Find the most recent results file if not specified
    if results_file is None:
        results_files = sorted(RESULTS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
        if not results_files:
            return {"error": "No results files found"}
        results_file = results_files[0]

    if verbose:
        print(f"  Analyzing: {results_file.name}")

    # Run compare.py
    compare_script = ANALYSIS_DIR / "compare.py"
    if compare_script.exists():
        result = subprocess.run(
            [sys.executable, str(compare_script), "--input", str(results_file), "--json"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            try:
                analysis_results["comparison"] = json.loads(result.stdout)
            except json.JSONDecodeError:
                analysis_results["comparison"] = {"raw": result.stdout}
        else:
            analysis_results["comparison"] = {"error": result.stderr}
    else:
        analysis_results["comparison"] = {"note": "compare.py not found"}

    # Run generate_tables.py
    tables_script = ANALYSIS_DIR / "generate_tables.py"
    if tables_script.exists():
        result = subprocess.run(
            [sys.executable, str(tables_script), "--input", str(results_file)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            analysis_results["tables"] = {"output": result.stdout, "status": "generated"}
        else:
            analysis_results["tables"] = {"error": result.stderr}
    else:
        analysis_results["tables"] = {"note": "generate_tables.py not found"}

    return analysis_results


def save_results(results: dict, prefix: str = "benchmark") -> Path:
    """Save results to JSON file."""
    RESULTS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    filepath = RESULTS_DIR / filename

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    return filepath


def print_summary(results: dict):
    """Print a summary of results."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    # Dependencies
    if "dependencies" in results:
        print("\nDependencies:")
        for dep, available in results["dependencies"].items():
            status = "OK" if available else "MISSING"
            print(f"  {dep}: {status}")

    # Workloads
    if "workloads" in results:
        print("\nWorkload Results:")
        workloads = results["workloads"]
        if isinstance(workloads, dict) and "workloads" in workloads:
            workloads = workloads["workloads"]
        for name, data in workloads.items():
            if isinstance(data, dict):
                if "error" in data:
                    print(f"  {name}: ERROR - {data['error'][:50]}")
                else:
                    lg = data.get("langgraph", {})
                    apxm = data.get("apxm", {})
                    if "mean_ms" in lg:
                        print(f"  {name}:")
                        print(f"    LangGraph: {lg['mean_ms']:.1f} ms")
                    if isinstance(apxm, dict) and "note" in apxm:
                        print(f"    A-PXM: {apxm['note']}")

    # Runtime
    if "runtime" in results:
        print("\nRuntime Benchmarks:")
        for name, data in results.get("runtime", {}).get("rust_benchmarks", {}).items():
            if "error" in data:
                print(f"  {name}: ERROR")
            elif "note" in data:
                print(f"  {name}: {data['note']}")
            else:
                print(f"  {name}: OK")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="A-PXM Benchmark Suite Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_all.py                    # Run everything
    python run_all.py --workloads        # Only DSL comparison workloads
    python run_all.py --runtime          # Only Rust runtime benchmarks
    python run_all.py --analyze          # Only analysis on existing results
    python run_all.py --quick            # Quick mode (fewer iterations)
    python run_all.py --workload 1,2,5   # Run specific workloads
        """,
    )
    parser.add_argument(
        "--workloads", action="store_true",
        help="Run only DSL comparison workloads"
    )
    parser.add_argument(
        "--runtime", action="store_true",
        help="Run only Rust runtime benchmarks"
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Run only analysis on existing results"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode with fewer iterations"
    )
    parser.add_argument(
        "--workload", type=str,
        help="Comma-separated list of workload numbers to run (e.g., 1,2,5)"
    )
    parser.add_argument(
        "--iterations", type=int,
        help="Override number of iterations"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output only JSON (no progress messages)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save results to file"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Determine iterations
    if args.iterations:
        iterations = args.iterations
    elif args.quick:
        iterations = 3
    else:
        iterations = config.get("llm", {}).get("iterations", 10)

    # Verbose mode (progress messages)
    verbose = not args.json

    # If no specific mode selected, run everything
    run_all = not (args.workloads or args.runtime or args.analyze)

    # Results container
    results = {
        "meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "suite": "apxm_benchmarks",
            "mode": "full" if run_all else "partial",
        },
        "config": {
            "iterations": iterations,
            "quick": args.quick,
        },
    }

    # Check dependencies
    if verbose:
        print("Checking dependencies...")
    results["dependencies"] = check_dependencies()

    # Parse workload numbers
    workload_nums = None
    if args.workload:
        try:
            workload_nums = [int(x.strip()) for x in args.workload.split(",")]
        except ValueError:
            print(f"Error: Invalid workload numbers: {args.workload}", file=sys.stderr)
            sys.exit(1)

    # Run workloads
    if run_all or args.workloads:
        if verbose:
            print("\nRunning DSL comparison workloads...")
        results["workloads"] = run_workloads(
            workload_nums=workload_nums,
            iterations=iterations,
            verbose=verbose,
        )

    # Run runtime benchmarks
    if run_all or args.runtime:
        if results["dependencies"].get("cargo"):
            if verbose:
                print("\nRunning Rust runtime benchmarks...")
            results["runtime"] = run_runtime_benchmarks(quick=args.quick, verbose=verbose)
        else:
            results["runtime"] = {"error": "Cargo not available"}

    # Save results before analysis
    results_file = None
    if not args.no_save and (run_all or args.workloads or args.runtime):
        results_file = save_results(results)
        if verbose:
            print(f"\nResults saved to: {results_file}")

    # Run analysis
    if run_all or args.analyze:
        if verbose:
            print("\nRunning analysis...")
        results["analysis"] = run_analysis(results_file, verbose=verbose)

    # Output
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_summary(results)

    # Return exit code based on errors
    has_errors = False
    if "error" in results.get("workloads", {}):
        has_errors = True
    if "error" in results.get("runtime", {}):
        has_errors = True

    sys.exit(1 if has_errors else 0)


if __name__ == "__main__":
    main()
