#!/usr/bin/env python3
"""
Print a simple ASCII latency breakdown for benchmark results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def render_bar(segments: List[Tuple[str, float]], width: int = 40) -> str:
    total = sum(v for _, v in segments if v > 0)
    if total <= 0:
        return ""

    bar = []
    used = 0
    for i, (label, value) in enumerate(segments):
        if value <= 0:
            continue
        if i == len(segments) - 1:
            seg_width = width - used
        else:
            seg_width = max(1, int(round(value / total * width)))
        bar.append(label * seg_width)
        used += seg_width
        if used >= width:
            break
    return "".join(bar)[:width].ljust(width)


def get_mean(metrics: Dict[str, Dict[str, float]], key: str) -> float:
    return float(metrics.get(key, {}).get("mean_ms", 0.0))


def format_ms(value: float) -> str:
    return f"{value:.1f} ms"


def summarize_langgraph(data: Dict[str, float]) -> Dict[str, float]:
    total = float(data.get("mean_ms", 0.0))
    llm = data.get("llm", {})
    llm_total = float(llm.get("total_ms_mean", 0.0))
    llm_wall = min(llm_total, total)
    non_llm = max(total - llm_wall, 0.0)
    return {
        "total": total,
        "llm_total": llm_total,
        "llm_wall": llm_wall,
        "non_llm": non_llm,
    }


def summarize_apxm(data: Dict[str, float]) -> Dict[str, float]:
    metrics = data.get("metrics", {})
    compile_ms = get_mean(metrics, "compile_ms")
    artifact_ms = get_mean(metrics, "artifact_ms")
    validation_ms = get_mean(metrics, "validation_ms")
    runtime_ms = get_mean(metrics, "runtime_ms")
    llm_total = get_mean(metrics, "llm_total_ms")

    total = float(data.get("mean_ms", 0.0))
    llm_wall = min(llm_total, runtime_ms if runtime_ms > 0 else total)
    runtime_non_llm = max((runtime_ms if runtime_ms > 0 else total) - llm_wall, 0.0)

    return {
        "total": total,
        "compile": compile_ms,
        "artifact": artifact_ms,
        "validation": validation_ms,
        "runtime_non_llm": runtime_non_llm,
        "llm_total": llm_total,
        "llm_wall": llm_wall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency breakdown from benchmark JSON")
    parser.add_argument("results", type=Path, help="Path to benchmark JSON")
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    workloads = data.get("workloads", {})

    if not workloads:
        print("No workloads found in results.")
        return

    print("Legend: C=compile A=artifact V=validation R=runtime(non-LLM) L=LLM N=non-LLM (LangGraph)")
    print()

    for name, result in workloads.items():
        print(f"{name}:")
        lg = result.get("langgraph", {})
        if lg and "error" not in lg:
            lg_summary = summarize_langgraph(lg)
            lg_bar = render_bar(
                [("L", lg_summary["llm_wall"]), ("N", lg_summary["non_llm"])],
                width=40,
            )
            print(f"  LangGraph total: {format_ms(lg_summary['total'])}")
            print(f"  [{lg_bar}] LLM (cum): {format_ms(lg_summary['llm_total'])}")

        apxm = result.get("apxm", {})
        if apxm and "error" not in apxm:
            apxm_summary = summarize_apxm(apxm)
            apxm_bar = render_bar(
                [
                    ("C", apxm_summary["compile"]),
                    ("A", apxm_summary["artifact"]),
                    ("V", apxm_summary["validation"]),
                    ("R", apxm_summary["runtime_non_llm"]),
                    ("L", apxm_summary["llm_wall"]),
                ],
                width=40,
            )
            print(f"  A-PXM total:     {format_ms(apxm_summary['total'])}")
            print(f"  [{apxm_bar}] LLM (cum): {format_ms(apxm_summary['llm_total'])}")
        print()


if __name__ == "__main__":
    main()
