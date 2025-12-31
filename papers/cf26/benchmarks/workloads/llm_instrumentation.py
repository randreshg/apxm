"""
LLM instrumentation helpers for LangGraph benchmarks.

Enforces real LLM usage (no mocks) and captures per-call latency.
"""

from __future__ import annotations

import statistics
import threading
import time
from typing import Dict, List

try:
    from langchain_ollama import ChatOllama
    HAS_OLLAMA = True
except ImportError:
    ChatOllama = None  # type: ignore[assignment]
    HAS_OLLAMA = False


_LOCK = threading.Lock()
_LATENCIES_MS: List[float] = []


def require_ollama() -> None:
    if not HAS_OLLAMA:
        raise RuntimeError(
            "langchain_ollama is required for benchmarks. Install it and run Ollama."
        )


def record_latency_ms(latency_ms: float) -> None:
    with _LOCK:
        _LATENCIES_MS.append(latency_ms)


def reset_metrics() -> None:
    with _LOCK:
        _LATENCIES_MS.clear()


def consume_latencies_ms() -> List[float]:
    with _LOCK:
        latencies = list(_LATENCIES_MS)
        _LATENCIES_MS.clear()
    return latencies


def summarize_latencies(latencies_ms: List[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {
            "count": 0,
            "total_ms": 0.0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p99_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }

    latencies_sorted = sorted(latencies_ms)
    count = len(latencies_sorted)
    total_ms = sum(latencies_sorted)
    p50_index = int(count * 0.5)
    p99_index = int(count * 0.99)

    return {
        "count": count,
        "total_ms": total_ms,
        "mean_ms": statistics.mean(latencies_sorted),
        "p50_ms": latencies_sorted[min(p50_index, count - 1)],
        "p99_ms": latencies_sorted[min(p99_index, count - 1)],
        "min_ms": latencies_sorted[0],
        "max_ms": latencies_sorted[-1],
    }


class TimedLLM:
    """Proxy that records per-call latency."""

    def __init__(self, llm) -> None:
        self._llm = llm

    def invoke(self, *args, **kwargs):
        start = time.perf_counter()
        result = self._llm.invoke(*args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        record_latency_ms(latency_ms)
        return result

    def __getattr__(self, name):
        return getattr(self._llm, name)


def get_ollama_llm(model: str, temperature: float = 0.0):
    require_ollama()
    return TimedLLM(ChatOllama(model=model, temperature=temperature))
