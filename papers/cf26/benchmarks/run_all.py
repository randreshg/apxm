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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Paths
BENCHMARKS_DIR = Path(__file__).parent


def get_conda_python() -> str:
    """Get the Python executable from the apxm conda environment.

    Returns the conda Python if available, otherwise falls back to sys.executable.
    """
    # Check CONDA_PREFIX environment variable first
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_python = Path(conda_prefix) / "bin" / "python"
        if conda_python.exists():
            return str(conda_python)

    # Try to find apxm conda environment
    try:
        result = subprocess.run(
            ["conda", "info", "--envs", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            info = json.loads(result.stdout)
            for env_path in info.get("envs", []):
                if "apxm" in env_path:
                    conda_python = Path(env_path) / "bin" / "python"
                    if conda_python.exists():
                        return str(conda_python)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass

    # Fall back to sys.executable
    return sys.executable


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


def find_apxm_config() -> Optional[Path]:
    """Locate .apxm/config.toml (project-scoped first, then user)."""
    cwd = Path.cwd()
    for ancestor in [cwd] + list(cwd.parents):
        candidate = ancestor / ".apxm" / "config.toml"
        if candidate.exists():
            return candidate
    home_candidate = Path.home() / ".apxm" / "config.toml"
    if home_candidate.exists():
        return home_candidate
    return None


def check_dependencies() -> dict:
    """Check availability of required dependencies."""
    deps = {
        "python": True,  # Obviously available
        "ollama": False,
        "apxm": False,
        "cargo": False,
        "langgraph": False,
        "langchain_ollama": False,
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

    # Check langchain-ollama
    try:
        import langchain_ollama
        deps["langchain_ollama"] = True
    except ImportError:
        pass

    return deps


def run_workloads(
    workload_nums: Optional[list] = None,
    iterations: int = 10,
    warmup: int = 3,
    verbose: bool = True,
) -> dict:
    """Run DSL comparison workloads."""
    runner_script = WORKLOADS_DIR / "runner.py"

    if not runner_script.exists():
        return {"error": f"Runner script not found: {runner_script}"}

    python_exe = get_conda_python()
    cmd = [python_exe, str(runner_script), "--json"]
    env = os.environ.copy()

    cfg = load_config()
    llm_cfg = (cfg.get("llm", {}) or {})
    provider = llm_cfg.get("provider")
    model = llm_cfg.get("model")
    base_url = llm_cfg.get("base_url")
    model_priority = llm_cfg.get("model_priority") or ([] if model is None else [model])

    apxm_config = find_apxm_config()

    if apxm_config is None:
        # Prefer the first locally-available model in model_priority, if `ollama` exists.
        if provider == "ollama" and model_priority:
            try:
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    available = set()
                    for line in result.stdout.splitlines():
                        line = line.strip()
                        if not line or line.upper().startswith("NAME"):
                            continue
                        available.add(line.split()[0])
                    for candidate in model_priority:
                        if candidate in available:
                            model = candidate
                            break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        if provider:
            env.setdefault("APXM_BENCH_PROVIDER", provider)
        if model:
            env.setdefault("APXM_BENCH_MODEL", model)
        if base_url:
            env.setdefault("APXM_BENCH_BASE_URL", base_url)
        if provider == "ollama" and model:
            env.setdefault("APXM_BENCH_OLLAMA_MODEL", model)
            env.setdefault("OLLAMA_MODEL", model)

    env["APXM_BENCH_ITERATIONS"] = str(iterations)
    env["APXM_BENCH_WARMUP"] = str(warmup)

    if workload_nums:
        # Run each specified workload
        all_results = {"workloads": {}}
        for num in workload_nums:
            if verbose:
                print(f"  Running workload {num}...")
            result = subprocess.run(
                cmd
                + ["--workload", str(num), "--iterations", str(iterations), "--warmup", str(warmup)],
                capture_output=True,
                text=True,
                cwd=WORKLOADS_DIR,
                env=env,
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
            cmd + ["--iterations", str(iterations), "--warmup", str(warmup)],
            capture_output=True,
            text=True,
            cwd=WORKLOADS_DIR,
            env=env,
        )
        if result.returncode == 0:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON output from runner"}
        return {"error": result.stderr or "Unknown error"}


def run_single_benchmark(benchmark_name: str, iterations: int = 5, verbose: bool = True) -> dict:
    """Run a single benchmark/workload by folder name.

    Args:
        benchmark_name: Folder name in workloads/ directory
        iterations: Number of iterations to run
        verbose: Print progress messages
    """
    # Special case: substrate_overhead is a Rust example, not a Python script
    if benchmark_name == "substrate_overhead":
        if verbose:
            print(f"  Running paper_benchmarks.rs (substrate overhead)...")
        try:
            crate_path = BENCHMARKS_DIR.parent.parent.parent / "crates" / "runtime" / "apxm-runtime"
            cmd = [
                "cargo", "run", "--release", "--features", "metrics",
                "--example", "paper_benchmarks",
                "-p", "apxm-runtime",
                "--", "--json",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=crate_path.parent.parent.parent,
                timeout=120,
            )
            if result.returncode == 0:
                start = result.stdout.find("{")
                end = result.stdout.rfind("}") + 1
                if start >= 0 and end > start:
                    return json.loads(result.stdout[start:end])
            return {"error": result.stderr[:500] if result.stderr else "Unknown error"}
        except subprocess.TimeoutExpired:
            return {"error": "Timeout (2 min)"}
        except Exception as e:
            return {"error": str(e)}

    # Normal workload: look for run.py in workloads/<name>/
    workload_dir = WORKLOADS_DIR / benchmark_name
    run_script = workload_dir / "run.py"

    if not workload_dir.exists():
        return {"error": f"Workload folder not found: {workload_dir}"}
    if not run_script.exists():
        return {"error": f"run.py not found in {workload_dir}"}

    if verbose:
        print(f"  Running {benchmark_name}...")

    try:
        cmd = [get_conda_python(), str(run_script), "--json", "--iterations", str(iterations)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=workload_dir,
        )

        if result.returncode == 0:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return {"raw_output": result.stdout[:2000], "note": "Could not parse JSON"}
        else:
            return {"error": result.stderr[:500] if result.stderr else "Unknown error"}
    except subprocess.TimeoutExpired:
        return {"error": "Timeout (5 min)"}
    except Exception as e:
        return {"error": str(e)}


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
            [get_conda_python(), str(compare_script), "--input", str(results_file), "--json"],
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
            [get_conda_python(), str(tables_script), "--input", str(results_file)],
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


def save_results(
    results: dict,
    prefix: str = "benchmark",
    output_dir: Optional[Path] = None,
) -> dict:
    """Save combined and per-workload results to JSON files."""
    RESULTS_DIR.mkdir(exist_ok=True)

    run_id = results.get("meta", {}).get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_filename = f"{prefix}_{run_id}.json"
    combined_path = RESULTS_DIR / combined_filename

    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)

    run_dir = output_dir or (RESULTS_DIR / f"run_{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "timestamp": results.get("meta", {}).get("timestamp"),
        "suite": results.get("meta", {}).get("suite"),
        "config": results.get("config", {}),
        "dependencies": results.get("dependencies", {}),
        "artifacts": {
            "suite": combined_filename,
        },
    }

    workloads_payload = results.get("workloads", {})
    workloads_meta = workloads_payload.get("meta") if isinstance(workloads_payload, dict) else None
    workloads_config = workloads_payload.get("config") if isinstance(workloads_payload, dict) else None
    workloads = workloads_payload.get("workloads") if isinstance(workloads_payload, dict) else None
    if isinstance(workloads, dict):
        workload_files = {}
        for name, data in workloads.items():
            filename = f"workload_{name}.json"
            path = run_dir / filename
            payload = {
                "meta": {
                    "run_id": run_id,
                    "timestamp": results.get("meta", {}).get("timestamp"),
                    "suite": results.get("meta", {}).get("suite"),
                    "workload": name,
                },
                "suite_config": results.get("config", {}),
                "workloads_meta": workloads_meta,
                "workloads_config": workloads_config,
                "result": data,
            }
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
            workload_files[name] = filename
        manifest["artifacts"]["workloads"] = workload_files

    benchmarks = results.get("benchmark")
    if isinstance(benchmarks, dict):
        benchmark_files = {}
        for name, data in benchmarks.items():
            filename = f"benchmark_{name}.json"
            path = run_dir / filename
            payload = {
                "meta": {
                    "run_id": run_id,
                    "timestamp": results.get("meta", {}).get("timestamp"),
                    "suite": results.get("meta", {}).get("suite"),
                    "benchmark": name,
                },
                "suite_config": results.get("config", {}),
                "result": data,
            }
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
            benchmark_files[name] = filename
        manifest["artifacts"]["benchmarks"] = benchmark_files

    runtime = results.get("runtime")
    if runtime:
        runtime_path = run_dir / "runtime.json"
        with open(runtime_path, "w") as f:
            json.dump(runtime, f, indent=2)
        manifest["artifacts"]["runtime"] = runtime_path.name

    analysis = results.get("analysis")
    if analysis:
        analysis_path = run_dir / "analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)
        manifest["artifacts"]["analysis"] = analysis_path.name

    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return {
        "combined_path": combined_path,
        "run_dir": run_dir,
        "manifest_path": manifest_path,
    }


def print_summary(results: dict):
    """Print a summary of results."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    def fmt_ms(value: Optional[float]) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.1f} ms"
        except (TypeError, ValueError):
            return "n/a"

    def fmt_num(value: Optional[float]) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.1f}"
        except (TypeError, ValueError):
            return "n/a"

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
                    print(f"  {name}:")
                    if "mean_ms" in lg:
                        llm = lg.get("llm", {}) if isinstance(lg.get("llm"), dict) else {}
                        tokens_in = llm.get("input_tokens_mean")
                        tokens_out = llm.get("output_tokens_mean")
                        token_text = ""
                        if tokens_in or tokens_out:
                            token_text = f", tokens {fmt_num(tokens_in)}/{fmt_num(tokens_out)}"
                        print(
                            "    LangGraph: "
                            f"mean {fmt_ms(lg.get('mean_ms'))}, "
                            f"p95 {fmt_ms(lg.get('p95_ms'))}, "
                            f"llm {fmt_ms(llm.get('total_ms_mean'))}, "
                            f"calls {fmt_num(llm.get('calls_mean'))}{token_text}"
                        )
                    elif "error" in lg:
                        print(f"    LangGraph: ERROR - {lg['error'][:50]}")

                    if isinstance(apxm, dict) and "mean_ms" in apxm:
                        metrics = apxm.get("metrics", {}) if isinstance(apxm.get("metrics"), dict) else {}
                        llm_total = metrics.get("llm_total_ms", {}) if isinstance(metrics.get("llm_total_ms"), dict) else {}
                        llm_requests = metrics.get("llm_requests", {}) if isinstance(metrics.get("llm_requests"), dict) else {}
                        llm_input = metrics.get("llm_input_tokens", {}) if isinstance(metrics.get("llm_input_tokens"), dict) else {}
                        llm_output = metrics.get("llm_output_tokens", {}) if isinstance(metrics.get("llm_output_tokens"), dict) else {}
                        token_text = ""
                        if llm_input or llm_output:
                            token_text = (
                                f", tokens {fmt_num(llm_input.get('mean_ms'))}/"
                                f"{fmt_num(llm_output.get('mean_ms'))}"
                            )
                        print(
                            "    A-PXM:      "
                            f"mean {fmt_ms(apxm.get('mean_ms'))}, "
                            f"p95 {fmt_ms(apxm.get('p95_ms'))}, "
                            f"llm {fmt_ms(llm_total.get('mean_ms'))}, "
                            f"calls {fmt_num(llm_requests.get('mean_ms'))}{token_text}"
                        )
                    elif isinstance(apxm, dict) and "note" in apxm:
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
        "--benchmark", type=str,
        help="Run a single workload by folder name (e.g., 1_parallel_research, parallel_research)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available benchmarks"
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
        "--warmup", type=int,
        help="Override warmup iterations"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output only JSON (no progress messages)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save results to file"
    )
    parser.add_argument(
        "--output-dir", type=str,
        help="Directory to store per-workload results (defaults to results/run_<timestamp>)"
    )
    parser.add_argument(
        "--tables", action="store_true",
        help="Auto-generate CSV tables after benchmark run"
    )

    args = parser.parse_args()

    # Handle --list: show available benchmarks and exit (use unified runner)
    if args.list:
        print("\nAll DSL Workloads (1-10):\n")
        # Use the unified runner to list all workloads
        result = subprocess.run(
            [get_conda_python(), str(WORKLOADS_DIR / "runner.py"), "--list"],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print("Additional Rust benchmarks:")
        print("  substrate_overhead (paper_benchmarks.rs)")
        print("\nUsage:")
        print("  python run_all.py --workload 1,5,10    # Run specific workloads")
        print("  python run_all.py --workloads          # Run all DSL workloads (1-10)")
        print("  python run_all.py --benchmark <name>   # Run by folder name")
        sys.exit(0)

    # Load configuration
    config = load_config()

    # Determine iterations
    if args.iterations:
        iterations = args.iterations
    elif args.quick:
        iterations = 3
    else:
        iterations = config.get("llm", {}).get("iterations", 10)

    # Determine warmup
    if args.warmup is not None:
        warmup = args.warmup
    elif args.quick:
        warmup = 1
    else:
        warmup = config.get("llm", {}).get("warmup", 3)

    # Verbose mode (progress messages)
    verbose = not args.json

    # If no specific mode selected, run everything
    run_all = not (args.workloads or args.runtime or args.benchmark or args.analyze)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Results container
    results = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "suite": "apxm_benchmarks",
            "mode": "full" if run_all else "partial",
        },
        "config": {
            "iterations": iterations,
            "warmup": warmup,
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
            warmup=warmup,
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

    # Run single benchmark by name
    if args.benchmark:
        if verbose:
            print(f"\nRunning benchmark: {args.benchmark}")
        results["benchmark"] = {
            args.benchmark: run_single_benchmark(
                args.benchmark,
                iterations=iterations,
                verbose=verbose
            )
        }

    # Save results before analysis
    results_file = None
    if not args.no_save and (run_all or args.workloads or args.runtime or args.benchmark):
        output_dir = Path(args.output_dir) if args.output_dir else None
        saved = save_results(results, output_dir=output_dir)
        results_file = saved.get("combined_path")
        run_dir = saved.get("run_dir")
        if verbose:
            print(f"\nResults saved to: {results_file}")
            print(f"Per-workload results: {run_dir}")

        # Auto-generate CSV tables if requested
        if args.tables and results_file:
            tables_script = ANALYSIS_DIR / "generate_tables.py"
            if tables_script.exists():
                tables_dir = run_dir / "tables" if run_dir else RESULTS_DIR / "tables"
                tables_dir.mkdir(parents=True, exist_ok=True)
                output_file = tables_dir / "summary.csv"

                if verbose:
                    print("\nGenerating CSV tables...")

                table_cmd = [
                    get_conda_python(),
                    str(tables_script),
                    "--input", str(results_file),
                    "--format", "csv",
                    "--output", str(output_file),
                ]
                table_result = subprocess.run(
                    table_cmd,
                    capture_output=True,
                    text=True,
                )
                if table_result.returncode == 0:
                    if verbose:
                        print(f"Tables saved to: {output_file}")
                    results["tables"] = {"output": str(output_file)}
                else:
                    if verbose:
                        print(f"Table generation failed: {table_result.stderr[:200]}")
                    results["tables"] = {"error": table_result.stderr[:500]}
            else:
                if verbose:
                    print("Warning: generate_tables.py not found")

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
