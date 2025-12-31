"""
Shared LangGraph benchmarking helpers with LLM latency breakdowns.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Dict, List

from llm_instrumentation import consume_latencies_ms, reset_metrics, summarize_latencies


def _summary_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "p50_ms": 0.0,
        }

    return {
        "mean_ms": statistics.mean(values),
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_ms": min(values),
        "max_ms": max(values),
        "p50_ms": statistics.median(values),
    }


def run_graph(graph, initial_state: Dict[str, Any], iterations: int, warmup: int) -> Dict[str, Any]:
    # Warmup (discard metrics)
    for _ in range(warmup):
        reset_metrics()
        graph.invoke(initial_state)
        consume_latencies_ms()

    samples: List[float] = []
    llm_summaries: List[Dict[str, float]] = []
    llm_totals: List[float] = []
    llm_calls: List[float] = []

    for _ in range(iterations):
        reset_metrics()
        start = time.perf_counter()
        graph.invoke(initial_state)
        elapsed_ms = (time.perf_counter() - start) * 1000
        samples.append(elapsed_ms)

        latencies = consume_latencies_ms()
        summary = summarize_latencies(latencies)
        llm_summaries.append(summary)
        llm_totals.append(summary["total_ms"])
        llm_calls.append(summary["count"])

    timing = _summary_stats(samples)
    llm_total_stats = _summary_stats(llm_totals)
    llm_calls_stats = _summary_stats(llm_calls)

    llm_total_mean = llm_total_stats["mean_ms"]
    wall_mean = timing["mean_ms"]
    llm_wall_mean = min(llm_total_mean, wall_mean)
    non_llm_mean = max(wall_mean - llm_wall_mean, 0.0)

    return {
        **timing,
        "samples": samples,
        "llm": {
            "total_ms_mean": llm_total_mean,
            "total_ms_std": llm_total_stats["std_ms"],
            "total_ms_min": llm_total_stats["min_ms"],
            "total_ms_max": llm_total_stats["max_ms"],
            "total_ms_p50": llm_total_stats["p50_ms"],
            "calls_mean": llm_calls_stats["mean_ms"],
            "calls_min": llm_calls_stats["min_ms"],
            "calls_max": llm_calls_stats["max_ms"],
            "wall_llm_ms_mean": llm_wall_mean,
            "non_llm_ms_mean": non_llm_mean,
        },
        "llm_samples": llm_summaries,
    }
