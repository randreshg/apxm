#!/usr/bin/env python3
"""
LangGraph benchmark utilities.

Runs a compiled graph with warmups, captures wall time, and aggregates LLM metrics.
"""

from __future__ import annotations

import io
import math
import statistics
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from llm_instrumentation import (
    consume_metrics,
    get_llm_settings,
    reset_metrics,
    summarize_calls,
    HAS_OLLAMA,
)


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    ordered = sorted(values)
    idx = (len(ordered) - 1) * (pct / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    return {
        "mean_ms": statistics.mean(values),
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_ms": min(values),
        "max_ms": max(values),
        "p50_ms": statistics.median(values),
        "p95_ms": _percentile(values, 95) or 0.0,
    }


def _truncate_value(value: Any, limit: int = 300) -> Any:
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return value[:limit] + "...[truncated]"
    if isinstance(value, dict):
        return {k: _truncate_value(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        trimmed = [_truncate_value(v, limit) for v in value[:10]]
        if len(value) > 10:
            trimmed.append("... [truncated]")
        return trimmed
    return value


def run_graph(
    graph: Any,
    initial_state: Dict[str, Any],
    iterations: int = 10,
    warmup: int = 3,
) -> Dict[str, Any]:
    """Run a LangGraph graph with instrumentation."""
    settings = get_llm_settings()

    # Warmup runs
    for _ in range(warmup):
        reset_metrics()
        try:
            graph.invoke(initial_state.copy())
        except Exception:
            pass
        consume_metrics()

    samples: List[float] = []
    sample_details: List[Dict[str, Any]] = []
    llm_samples: List[Dict[str, float]] = []
    llm_total_ms: List[float] = []
    llm_calls: List[float] = []
    llm_input_tokens: List[float] = []
    llm_output_tokens: List[float] = []
    llm_wall_ms: List[float] = []
    non_llm_ms: List[float] = []

    for i in range(iterations):
        reset_metrics()
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        start = time.perf_counter()
        success = True
        error = None
        result = None

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                result = graph.invoke(initial_state.copy())
        except Exception as exc:  # pragma: no cover - runtime errors are reported
            success = False
            error = str(exc)
        wall_time_ms = (time.perf_counter() - start) * 1000.0

        calls = consume_metrics()
        call_summary = summarize_calls(calls)
        llm_samples.append(call_summary)

        sample_entry = {
            "iteration": i,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": success,
            "wall_time_ms": wall_time_ms,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "llm": call_summary,
        }
        if error:
            sample_entry["error"] = error
        if result is not None:
            sample_entry["result_summary"] = _truncate_value(result)
        sample_details.append(sample_entry)

        if success:
            samples.append(wall_time_ms)
            total_ms = float(call_summary.get("total_ms", 0.0))
            count = float(call_summary.get("count", 0.0))
            llm_total_ms.append(total_ms)
            llm_calls.append(count)
            llm_input_tokens.append(float(call_summary.get("input_tokens_total", 0.0)))
            llm_output_tokens.append(float(call_summary.get("output_tokens_total", 0.0)))

            llm_wall = min(total_ms, wall_time_ms)
            llm_wall_ms.append(llm_wall)
            non_llm_ms.append(max(wall_time_ms - llm_wall, 0.0))

    if not samples:
        return {
            "error": "No successful runs",
            "samples": [],
            "sample_details": sample_details,
            "llm_samples": llm_samples,
            "has_ollama": HAS_OLLAMA,
        }

    llm_summary = {
        "total_ms_mean": statistics.mean(llm_total_ms) if llm_total_ms else 0.0,
        "total_ms_std": statistics.stdev(llm_total_ms) if len(llm_total_ms) > 1 else 0.0,
        "total_ms_min": min(llm_total_ms) if llm_total_ms else 0.0,
        "total_ms_max": max(llm_total_ms) if llm_total_ms else 0.0,
        "total_ms_p50": statistics.median(llm_total_ms) if llm_total_ms else 0.0,
        "total_ms_p95": _percentile(llm_total_ms, 95) or 0.0,
        "calls_mean": statistics.mean(llm_calls) if llm_calls else 0.0,
        "calls_min": min(llm_calls) if llm_calls else 0.0,
        "calls_max": max(llm_calls) if llm_calls else 0.0,
        "input_tokens_mean": statistics.mean(llm_input_tokens) if llm_input_tokens else 0.0,
        "output_tokens_mean": statistics.mean(llm_output_tokens) if llm_output_tokens else 0.0,
        "wall_llm_ms_mean": statistics.mean(llm_wall_ms) if llm_wall_ms else 0.0,
        "non_llm_ms_mean": statistics.mean(non_llm_ms) if non_llm_ms else 0.0,
    }

    return {
        **_stats(samples),
        "samples": samples,
        "iterations": iterations,
        "llm": llm_summary,
        "llm_samples": llm_samples,
        "sample_details": sample_details,
        "llm_provider": settings.provider,
        "llm_model": settings.model,
        "llm_base_url": settings.base_url,
        "llm_backend": settings.backend_name,
        "llm_config_source": settings.source,
        "has_ollama": HAS_OLLAMA,
    }
