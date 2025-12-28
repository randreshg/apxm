#!/usr/bin/env python3
"""
Benchmark Comparison Analysis

Processes JSON benchmark results and generates comparison metrics.

Usage:
    python compare.py --input results/benchmark_20251226.json --json
    python compare.py --input results/benchmark_20251226.json --markdown
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional


def calculate_speedup(baseline_ms: float, optimized_ms: float) -> float:
    """Calculate speedup ratio."""
    if optimized_ms <= 0:
        return float("inf")
    return baseline_ms / optimized_ms


def calculate_efficiency(actual_speedup: float, theoretical_max: float) -> float:
    """Calculate parallelism efficiency as percentage."""
    if theoretical_max <= 0:
        return 0.0
    return (actual_speedup / theoretical_max) * 100


def analyze_parallel_research(data: dict) -> dict:
    """Analyze parallel research workload results."""
    lg = data.get("langgraph", {})
    apxm = data.get("apxm", {})

    result = {
        "workload": "parallel_research",
        "langgraph": {},
        "apxm": {},
        "comparison": {},
    }

    # Extract LangGraph metrics
    if "mean_ms" in lg:
        result["langgraph"] = {
            "mean_ms": lg["mean_ms"],
            "std_ms": lg.get("std_ms", 0),
            "p50_ms": lg.get("p50_ms", lg["mean_ms"]),
        }

    # Extract A-PXM metrics
    if "execution" in apxm and isinstance(apxm["execution"], dict):
        exec_data = apxm["execution"]
        result["apxm"] = {
            "compile_time_ms": apxm.get("compile_time_ms", 0),
            "mean_ms": exec_data.get("mean_ms", 0),
            "std_ms": exec_data.get("std_ms", 0),
            "p50_ms": exec_data.get("p50_ms", 0),
        }
    elif "note" in apxm:
        result["apxm"]["note"] = apxm["note"]

    # Calculate comparison metrics
    if result["langgraph"].get("mean_ms") and result["apxm"].get("mean_ms"):
        lg_mean = result["langgraph"]["mean_ms"]
        apxm_mean = result["apxm"]["mean_ms"]
        result["comparison"] = {
            "speedup": calculate_speedup(lg_mean, apxm_mean),
            "latency_reduction_pct": ((lg_mean - apxm_mean) / lg_mean) * 100,
        }

    return result


def analyze_chain_fusion(data: dict) -> dict:
    """Analyze chain fusion workload (compiler optimization demo)."""
    lg = data.get("langgraph", {})
    apxm = data.get("apxm", {})

    result = {
        "workload": "chain_fusion",
        "description": "FuseReasoning: N sequential RSN calls -> 1 batched call",
        "langgraph": {},
        "apxm": {},
        "comparison": {},
    }

    chain_length = 5  # Default chain length

    # LangGraph: N separate calls
    if "mean_ms" in lg:
        result["langgraph"] = {
            "llm_calls": chain_length,
            "mean_ms": lg["mean_ms"],
            "estimated_per_call_ms": lg["mean_ms"] / chain_length,
        }

    # A-PXM: Fused into 1 call
    if "execution" in apxm:
        exec_data = apxm.get("execution", {})
        result["apxm"] = {
            "llm_calls": 1,  # Fused
            "mean_ms": exec_data.get("mean_ms", 0),
            "fusion_enabled": True,
        }
    elif "note" in apxm:
        result["apxm"]["note"] = apxm["note"]

    # Comparison
    if result["langgraph"].get("mean_ms") and result["apxm"].get("mean_ms"):
        result["comparison"] = {
            "llm_call_reduction": f"{chain_length}x -> 1x",
            "speedup": calculate_speedup(
                result["langgraph"]["mean_ms"],
                result["apxm"]["mean_ms"],
            ),
            "theoretical_speedup": chain_length,
        }

    return result


def analyze_scalability(data: dict) -> dict:
    """Analyze N-way parallelism scalability."""
    result = {
        "workload": "scalability",
        "levels": [],
        "comparison": {},
    }

    # Expected format: results for N=1,2,4,8
    for n in [1, 2, 4, 8]:
        level_data = data.get(f"parallel_{n}", {})
        if level_data:
            level = {
                "n": n,
                "theoretical_speedup": n,
            }
            if "mean_ms" in level_data.get("langgraph", {}):
                lg = level_data["langgraph"]
                level["langgraph_ms"] = lg["mean_ms"]
            if "mean_ms" in level_data.get("apxm", {}).get("execution", {}):
                apxm = level_data["apxm"]["execution"]
                level["apxm_ms"] = apxm["mean_ms"]
            result["levels"].append(level)

    # Calculate efficiency at each level
    if result["levels"]:
        baseline = result["levels"][0]  # N=1
        for level in result["levels"]:
            n = level["n"]
            if "langgraph_ms" in baseline and "langgraph_ms" in level:
                actual = calculate_speedup(baseline["langgraph_ms"], level["langgraph_ms"])
                level["langgraph_actual_speedup"] = actual
                level["langgraph_efficiency_pct"] = calculate_efficiency(actual, n)
            if "apxm_ms" in baseline and "apxm_ms" in level:
                actual = calculate_speedup(baseline["apxm_ms"], level["apxm_ms"])
                level["apxm_actual_speedup"] = actual
                level["apxm_efficiency_pct"] = calculate_efficiency(actual, n)

    return result


def analyze_type_verification(data: dict) -> dict:
    """Analyze compile-time vs runtime error detection."""
    result = {
        "workload": "type_verification",
        "description": "Compile-time error detection vs runtime failure",
        "langgraph": {},
        "apxm": {},
        "comparison": {},
    }

    lg = data.get("langgraph", {})
    apxm = data.get("apxm", {})

    # LangGraph: Runtime error
    if "error_detected_at" in lg:
        result["langgraph"] = {
            "error_type": "runtime",
            "time_to_error_ms": lg.get("time_to_error_ms", 0),
            "llm_calls_before_error": lg.get("llm_calls_before_error", 0),
            "cost_wasted_usd": lg.get("cost_wasted_usd", 0),
        }

    # A-PXM: Compile-time error
    if "compile_error" in apxm:
        result["apxm"] = {
            "error_type": "compile_time",
            "time_to_error_ms": apxm.get("compile_time_ms", 50),
            "llm_calls_before_error": 0,
            "cost_wasted_usd": 0,
        }
    elif "note" in apxm:
        result["apxm"]["note"] = apxm["note"]

    # Comparison
    lg_time = result["langgraph"].get("time_to_error_ms", 0)
    apxm_time = result["apxm"].get("time_to_error_ms", 50)
    if lg_time and apxm_time:
        result["comparison"] = {
            "time_saved_ms": lg_time - apxm_time,
            "cost_saved_usd": result["langgraph"].get("cost_wasted_usd", 0),
            "detection_speedup": calculate_speedup(lg_time, apxm_time),
        }

    return result


def analyze_workload(name: str, data: dict) -> dict:
    """Route to appropriate analyzer based on workload name."""
    analyzers = {
        "parallel_research": analyze_parallel_research,
        "chain_fusion": analyze_chain_fusion,
        "scalability": analyze_scalability,
        "type_verification": analyze_type_verification,
    }

    if name in analyzers:
        return analyzers[name](data)

    # Generic analysis for other workloads
    return {
        "workload": name,
        "langgraph": data.get("langgraph", {}),
        "apxm": data.get("apxm", {}),
    }


def analyze_results(results: dict) -> dict:
    """Analyze all benchmark results."""
    analysis = {
        "meta": results.get("meta", {}),
        "summary": {},
        "workloads": {},
    }

    # Find workloads data
    workloads = results.get("workloads", {})
    if isinstance(workloads, dict) and "workloads" in workloads:
        workloads = workloads["workloads"]

    # Analyze each workload
    for name, data in workloads.items():
        if isinstance(data, dict) and "error" not in data:
            analysis["workloads"][name] = analyze_workload(name, data)

    # Generate summary
    total_speedup = []
    total_efficiency = []

    for name, workload_analysis in analysis["workloads"].items():
        comparison = workload_analysis.get("comparison", {})
        if "speedup" in comparison:
            total_speedup.append(comparison["speedup"])
        if "levels" in workload_analysis:
            for level in workload_analysis["levels"]:
                if "apxm_efficiency_pct" in level:
                    total_efficiency.append(level["apxm_efficiency_pct"])

    if total_speedup:
        analysis["summary"]["average_speedup"] = sum(total_speedup) / len(total_speedup)
        analysis["summary"]["max_speedup"] = max(total_speedup)
    if total_efficiency:
        analysis["summary"]["average_efficiency_pct"] = sum(total_efficiency) / len(total_efficiency)

    return analysis


def format_markdown(analysis: dict) -> str:
    """Format analysis as markdown."""
    lines = []
    lines.append("# A-PXM vs LangGraph Comparison Results")
    lines.append("")
    lines.append(f"Generated: {analysis.get('meta', {}).get('timestamp', 'N/A')}")
    lines.append("")

    # Summary
    summary = analysis.get("summary", {})
    if summary:
        lines.append("## Summary")
        lines.append("")
        if "average_speedup" in summary:
            lines.append(f"- Average Speedup: **{summary['average_speedup']:.2f}x**")
        if "max_speedup" in summary:
            lines.append(f"- Max Speedup: **{summary['max_speedup']:.2f}x**")
        if "average_efficiency_pct" in summary:
            lines.append(f"- Average Efficiency: **{summary['average_efficiency_pct']:.1f}%**")
        lines.append("")

    # Workload details
    lines.append("## Workload Results")
    lines.append("")

    for name, data in analysis.get("workloads", {}).items():
        lines.append(f"### {name.replace('_', ' ').title()}")
        lines.append("")

        if "description" in data:
            lines.append(f"*{data['description']}*")
            lines.append("")

        # Create comparison table
        lines.append("| Metric | LangGraph | A-PXM |")
        lines.append("|--------|-----------|-------|")

        lg = data.get("langgraph", {})
        apxm = data.get("apxm", {})

        if "mean_ms" in lg:
            lg_val = f"{lg['mean_ms']:.1f} ms"
            apxm_val = f"{apxm.get('mean_ms', 'N/A')} ms" if "mean_ms" in apxm else apxm.get("note", "N/A")
            lines.append(f"| Mean Latency | {lg_val} | {apxm_val} |")

        if "llm_calls" in lg:
            lines.append(f"| LLM Calls | {lg['llm_calls']} | {apxm.get('llm_calls', 'N/A')} |")

        comparison = data.get("comparison", {})
        if "speedup" in comparison:
            lines.append(f"| **Speedup** | - | **{comparison['speedup']:.2f}x** |")

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSON file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--markdown", action="store_true", help="Output as Markdown")
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    args = parser.parse_args()

    # Load results
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path) as f:
        results = json.load(f)

    # Analyze
    analysis = analyze_results(results)

    # Output
    if args.markdown:
        output = format_markdown(analysis)
    else:
        output = json.dumps(analysis, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Analysis saved to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
