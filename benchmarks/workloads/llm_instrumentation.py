#!/usr/bin/env python3
"""
Lightweight LLM instrumentation for LangGraph benchmarks.

Captures per-call latency and token counts when using ChatOllama.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


try:
    from langchain_ollama import ChatOllama
    HAS_OLLAMA = True
except ImportError:  # pragma: no cover - optional dependency
    ChatOllama = None
    HAS_OLLAMA = False

try:
    from langchain_openai import ChatOpenAI
    HAS_OPENAI = True
except ImportError:  # pragma: no cover - optional dependency
    ChatOpenAI = None
    HAS_OPENAI = False


_LOCK = threading.Lock()
_CALLS: List[Dict[str, Optional[float]]] = []


@dataclass
class LLMSettings:
    provider: Optional[str]
    model: Optional[str]
    base_url: Optional[str]
    api_key: Optional[str]
    backend_name: Optional[str]
    options: Dict[str, Any]
    source: str


def _find_apxm_config() -> Optional[Path]:
    cwd = Path.cwd()
    for ancestor in [cwd] + list(cwd.parents):
        candidate = ancestor / ".apxm" / "config.toml"
        if candidate.exists():
            return candidate

    home = Path.home() / ".apxm" / "config.toml"
    if home.exists():
        return home
    return None


def _load_apxm_config() -> Optional[Dict[str, Any]]:
    path = _find_apxm_config()
    if not path:
        return None
    try:
        import tomllib
        return tomllib.loads(path.read_text())
    except Exception:
        return None


def _load_benchmark_config() -> Optional[Dict[str, Any]]:
    cfg_path = Path(__file__).resolve().parent.parent / "config.json"
    if not cfg_path.exists():
        return None
    try:
        return json.loads(cfg_path.read_text())
    except Exception:
        return None


def _resolve_api_key(raw_key: Optional[str], provider: Optional[str]) -> Optional[str]:
    if not raw_key:
        if provider and provider.lower() == "ollama":
            return ""
        return None
    if raw_key.startswith("env:"):
        env_name = raw_key[len("env:") :]
        return os.environ.get(env_name)
    return raw_key


def get_llm_settings() -> LLMSettings:
    provider_override = os.environ.get("APXM_BENCH_PROVIDER")
    model_override = os.environ.get("APXM_BENCH_MODEL")
    base_url_override = os.environ.get("APXM_BENCH_BASE_URL")
    api_key_override = os.environ.get("APXM_BENCH_API_KEY")
    backend_override = os.environ.get("APXM_BENCH_BACKEND")

    provider = provider_override
    model = model_override
    base_url = base_url_override
    api_key = api_key_override
    backend_name = backend_override
    options: Dict[str, Any] = {}
    source = "env"

    apxm_config = _load_apxm_config()
    if apxm_config:
        chat = apxm_config.get("chat", {}) if isinstance(apxm_config.get("chat"), dict) else {}
        backends = apxm_config.get("llm_backends", []) or []

        if not backend_name:
            providers = chat.get("providers", []) if isinstance(chat.get("providers", []), list) else []
            if providers:
                backend_name = providers[0]

        backend = None
        if backend_name:
            for entry in backends:
                if isinstance(entry, dict) and entry.get("name") == backend_name:
                    backend = entry
                    break
        if backend is None and backends:
            backend = backends[0] if isinstance(backends[0], dict) else None

        if backend:
            provider = provider or backend.get("provider") or "openai"
            model = model or backend.get("model")
            base_url = base_url or backend.get("endpoint")
            api_key = api_key or _resolve_api_key(backend.get("api_key"), provider)
            options = backend.get("options", {}) if isinstance(backend.get("options", {}), dict) else {}
            source = "apxm_config"

    bench_config = _load_benchmark_config()
    if apxm_config is None and bench_config:
        llm_cfg = bench_config.get("llm", {}) if isinstance(bench_config.get("llm"), dict) else {}
        provider = provider or llm_cfg.get("provider")
        model = model or llm_cfg.get("model")
        base_url = base_url or llm_cfg.get("base_url")
        if source == "env" and (provider or model or base_url):
            source = "bench_config"

    if provider and provider.lower() in {"vllm"}:
        provider = "openai"

    if not provider:
        provider = "ollama"

    if provider.lower() == "ollama" and not model:
        model = os.environ.get("APXM_BENCH_OLLAMA_MODEL") or os.environ.get("OLLAMA_MODEL")

    return LLMSettings(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        backend_name=backend_name,
        options=options,
        source=source,
    )


@dataclass
class BenchmarkSettings:
    """Benchmark-specific settings from ~/.apxm/config.toml [benchmarks] section."""
    timeout_seconds: float = 120.0
    iterations: int = 10
    warmup: int = 3


def get_benchmark_settings() -> BenchmarkSettings:
    """Load benchmark settings from ~/.apxm/config.toml [benchmarks] section.
    
    Example config:
        [benchmarks]
        timeout_seconds = 300.0
        iterations = 10
        warmup = 3
    
    Environment variables (APXM_BENCH_TIMEOUT, etc.) take precedence.
    """
    settings = BenchmarkSettings()
    
    apxm_config = _load_apxm_config()
    if apxm_config:
        bench_cfg = apxm_config.get("benchmarks", {})
        if isinstance(bench_cfg, dict):
            if "timeout_seconds" in bench_cfg:
                try:
                    settings.timeout_seconds = float(bench_cfg["timeout_seconds"])
                except (TypeError, ValueError):
                    pass
            if "iterations" in bench_cfg:
                try:
                    settings.iterations = int(bench_cfg["iterations"])
                except (TypeError, ValueError):
                    pass
            if "warmup" in bench_cfg:
                try:
                    settings.warmup = int(bench_cfg["warmup"])
                except (TypeError, ValueError):
                    pass
    
    # Environment variables take precedence
    timeout_env = os.environ.get("APXM_BENCH_TIMEOUT")
    if timeout_env:
        try:
            settings.timeout_seconds = float(timeout_env)
        except ValueError:
            pass
    
    iterations_env = os.environ.get("APXM_BENCH_ITERATIONS")
    if iterations_env:
        try:
            settings.iterations = int(iterations_env)
        except ValueError:
            pass
    
    warmup_env = os.environ.get("APXM_BENCH_WARMUP")
    if warmup_env:
        try:
            settings.warmup = int(warmup_env)
        except ValueError:
            pass
    
    return settings


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


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_tokens(response: Any) -> Tuple[Optional[int], Optional[int]]:
    if response is None:
        return None, None

    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        in_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
        out_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
        return _safe_int(in_tokens), _safe_int(out_tokens)

    response_meta = getattr(response, "response_metadata", None)
    if isinstance(response_meta, dict):
        in_tokens = response_meta.get("prompt_eval_count") or response_meta.get("input_tokens")
        out_tokens = response_meta.get("eval_count") or response_meta.get("output_tokens")
        return _safe_int(in_tokens), _safe_int(out_tokens)

    metadata = getattr(response, "metadata", None)
    if isinstance(metadata, dict):
        in_tokens = metadata.get("prompt_eval_count") or metadata.get("input_tokens")
        out_tokens = metadata.get("eval_count") or metadata.get("output_tokens")
        return _safe_int(in_tokens), _safe_int(out_tokens)

    return None, None


def reset_metrics() -> None:
    """Clear all recorded LLM call metrics."""
    with _LOCK:
        _CALLS.clear()


def record_call(latency_ms: float, input_tokens: Optional[int], output_tokens: Optional[int]) -> None:
    """Record a single LLM call."""
    with _LOCK:
        _CALLS.append(
            {
                "latency_ms": float(latency_ms),
                "input_tokens": float(input_tokens) if input_tokens is not None else None,
                "output_tokens": float(output_tokens) if output_tokens is not None else None,
                "total_tokens": float(input_tokens + output_tokens)
                if input_tokens is not None and output_tokens is not None
                else None,
            }
        )


def consume_metrics() -> List[Dict[str, Optional[float]]]:
    """Return and clear recorded metrics."""
    with _LOCK:
        calls = list(_CALLS)
        _CALLS.clear()
    return calls


def consume_latencies_ms() -> List[float]:
    """Return and clear recorded latencies."""
    calls = consume_metrics()
    return [c["latency_ms"] for c in calls if c.get("latency_ms") is not None]


def summarize_latencies(latencies_ms: List[float]) -> Dict[str, float]:
    """Summarize a list of latencies."""
    if not latencies_ms:
        return {"count": 0}

    mean_ms = sum(latencies_ms) / len(latencies_ms)
    if len(latencies_ms) > 1:
        variance = sum((v - mean_ms) ** 2 for v in latencies_ms) / (len(latencies_ms) - 1)
        std_ms = math.sqrt(variance)
    else:
        std_ms = 0.0

    return {
        "count": len(latencies_ms),
        "total_ms": sum(latencies_ms),
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "p50_ms": _percentile(latencies_ms, 50) or 0.0,
        "p95_ms": _percentile(latencies_ms, 95) or 0.0,
        "p99_ms": _percentile(latencies_ms, 99) or 0.0,
    }


def summarize_calls(calls: List[Dict[str, Optional[float]]]) -> Dict[str, float]:
    """Summarize call metrics (latency + tokens)."""
    latencies = [c["latency_ms"] for c in calls if c.get("latency_ms") is not None]
    input_tokens = [c["input_tokens"] for c in calls if c.get("input_tokens") is not None]
    output_tokens = [c["output_tokens"] for c in calls if c.get("output_tokens") is not None]

    summary = summarize_latencies(latencies)
    summary["input_tokens_total"] = float(sum(input_tokens)) if input_tokens else 0.0
    summary["output_tokens_total"] = float(sum(output_tokens)) if output_tokens else 0.0
    summary["total_tokens_total"] = (
        float(sum(input_tokens) + sum(output_tokens)) if input_tokens and output_tokens else 0.0
    )
    return summary


class _InstrumentedLLM:
    def __init__(self, llm: Any):
        self._llm = llm

    def invoke(self, prompt: Any, **kwargs: Any) -> Any:
        response = None
        start = time.perf_counter()
        try:
            response = self._llm.invoke(prompt, **kwargs)
            return response
        finally:
            latency_ms = (time.perf_counter() - start) * 1000.0
            input_tokens, output_tokens = _extract_tokens(response)
            record_call(latency_ms, input_tokens, output_tokens)

    async def ainvoke(self, prompt: Any, **kwargs: Any) -> Any:
        response = None
        start = time.perf_counter()
        try:
            response = await self._llm.ainvoke(prompt, **kwargs)
            return response
        finally:
            latency_ms = (time.perf_counter() - start) * 1000.0
            input_tokens, output_tokens = _extract_tokens(response)
            record_call(latency_ms, input_tokens, output_tokens)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


_LLM_CACHE: Dict[Tuple[str, str, Tuple[Tuple[str, Any], ...]], Any] = {}


def _init_llm(cls: Any, **kwargs: Any) -> Any:
    try:
        sig = inspect.signature(cls)
        if "base_url" in kwargs and "base_url" not in sig.parameters and "openai_api_base" in sig.parameters:
            kwargs["openai_api_base"] = kwargs.pop("base_url")
        if "api_key" in kwargs and "api_key" not in sig.parameters and "openai_api_key" in sig.parameters:
            kwargs["openai_api_key"] = kwargs.pop("api_key")
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
        return cls(**filtered)
    except (TypeError, ValueError):
        return cls(**{k: v for k, v in kwargs.items() if v is not None})


def get_ollama_llm(model: str, **kwargs: Any) -> Any:
    """Return an instrumented ChatOllama instance."""
    if not HAS_OLLAMA or ChatOllama is None:
        raise RuntimeError("langchain-ollama is required for Ollama benchmarks")

    key = ("ollama", model, tuple(sorted(kwargs.items())))
    llm = _LLM_CACHE.get(key)
    if llm is None:
        llm = _init_llm(ChatOllama, model=model, **kwargs)
        _LLM_CACHE[key] = llm
    return _InstrumentedLLM(llm)


def get_openai_llm(model: str, **kwargs: Any) -> Any:
    """Return an instrumented ChatOpenAI instance."""
    if not HAS_OPENAI or ChatOpenAI is None:
        raise RuntimeError("langchain-openai is required for OpenAI/vLLM benchmarks")

    key = ("openai", model, tuple(sorted(kwargs.items())))
    llm = _LLM_CACHE.get(key)
    if llm is None:
        llm = _init_llm(ChatOpenAI, model=model, **kwargs)
        _LLM_CACHE[key] = llm
    return _InstrumentedLLM(llm)


def get_llm() -> Any:
    """Return an instrumented LLM based on APxM config + env overrides."""
    settings = get_llm_settings()
    provider = (settings.provider or "ollama").lower()
    model = settings.model or ("phi3:mini" if provider == "ollama" else "gpt-4o-mini")
    options = settings.options.copy()

    if provider == "ollama":
        if settings.base_url:
            options.setdefault("base_url", settings.base_url)
        return get_ollama_llm(model, **options)

    if provider == "openai":
        if settings.base_url:
            options.setdefault("base_url", settings.base_url)
        if settings.api_key:
            options.setdefault("api_key", settings.api_key)
        return get_openai_llm(model, **options)

    raise RuntimeError(f"Unsupported LLM provider '{provider}'")
