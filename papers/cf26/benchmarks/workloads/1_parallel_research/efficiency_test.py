#!/usr/bin/env python3
"""
Parallelism Efficiency Benchmark

Measures actual parallelism efficiency by comparing:
- Sequential execution: Run 3 RSN ops one after another
- Parallel execution: Run 3 RSN ops concurrently

Efficiency = (sequential_time / 3) / parallel_time * 100%

Paper claim: "~85% parallelism efficiency"
"""

import asyncio
import json
import os
import statistics
import time
from datetime import datetime, timezone

from llm_instrumentation import get_ollama_llm, HAS_OLLAMA

OLLAMA_MODEL = (
    os.environ.get("APXM_BENCH_OLLAMA_MODEL")
    or os.environ.get("OLLAMA_MODEL")
    or "phi3:mini"
)


def _get_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


WARMUP_ITERATIONS = _get_int_env("APXM_BENCH_WARMUP", 3)
BENCHMARK_ITERATIONS = _get_int_env("APXM_BENCH_ITERATIONS", 10)

# Prompts for the 3 parallel research tasks
PROMPTS = [
    "Explain the background of quantum computing in 2 sentences.",
    "What are recent advances in quantum computing? Answer in 2 sentences.",
    "What is the societal impact of quantum computing? Answer in 2 sentences.",
]


def get_llm():
    """Get LLM instance."""
    return get_ollama_llm(OLLAMA_MODEL)


def run_sequential(prompts: list[str]) -> tuple[float, list[str]]:
    """Run prompts sequentially, return total time and responses."""
    llm = get_llm()

    start = time.perf_counter()
    responses = []
    for prompt in prompts:
        response = llm.invoke(prompt)
        responses.append(response.content)
    elapsed = time.perf_counter() - start
    return elapsed, responses


async def run_parallel_async(prompts: list[str]) -> tuple[float, list[str]]:
    """Run prompts in parallel using asyncio."""
    llm = get_llm()

    async def call_llm(prompt):
        # LangChain's invoke is sync, run in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, llm.invoke, prompt)

    start = time.perf_counter()
    tasks = [call_llm(p) for p in prompts]
    responses = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start

    return elapsed, [r.content for r in responses]


def run_parallel(prompts: list[str]) -> tuple[float, list[str]]:
    """Run prompts in parallel."""
    return asyncio.run(run_parallel_async(prompts))


def calculate_efficiency(sequential_time: float, parallel_time: float, n_ops: int) -> float:
    """Calculate parallelism efficiency.

    Efficiency = (sequential_time / n_ops) / parallel_time * 100
    Perfect parallelism would give 100%
    """
    theoretical_parallel = sequential_time / n_ops
    if parallel_time <= 0:
        return 0
    return (theoretical_parallel / parallel_time) * 100


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Parallelism Efficiency Benchmark")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    parser.add_argument("--iterations", type=int, default=BENCHMARK_ITERATIONS)
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERATIONS)
    args = parser.parse_args()

    print("=" * 60)
    print("PARALLELISM EFFICIENCY BENCHMARK")
    print("=" * 60)
    print(f"Ollama available: {HAS_OLLAMA}")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Prompts: {len(PROMPTS)}")
    print(f"Iterations: {args.iterations}")
    print()

    # Warmup
    print("Warming up...")
    for _ in range(args.warmup):
        run_sequential(PROMPTS[:1])  # Just one prompt for warmup
        run_parallel(PROMPTS[:1])

    # Benchmark sequential
    print("\nRunning sequential benchmark...")
    sequential_times = []
    for i in range(args.iterations):
        elapsed, _ = run_sequential(PROMPTS)
        sequential_times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.2f}s")

    # Benchmark parallel
    print("\nRunning parallel benchmark...")
    parallel_times = []
    for i in range(args.iterations):
        elapsed, _ = run_parallel(PROMPTS)
        parallel_times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.2f}s")

    # Calculate statistics
    seq_mean = statistics.mean(sequential_times)
    seq_std = statistics.stdev(sequential_times) if len(sequential_times) > 1 else 0
    par_mean = statistics.mean(parallel_times)
    par_std = statistics.stdev(parallel_times) if len(parallel_times) > 1 else 0

    # Calculate efficiency for each iteration pair
    efficiencies = [
        calculate_efficiency(seq, par, len(PROMPTS))
        for seq, par in zip(sequential_times, parallel_times)
    ]
    eff_mean = statistics.mean(efficiencies)
    eff_std = statistics.stdev(efficiencies) if len(efficiencies) > 1 else 0

    # Speedup
    speedup = seq_mean / par_mean if par_mean > 0 else 0
    theoretical_speedup = len(PROMPTS)

    results = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmark": "parallelism_efficiency",
            "has_ollama": HAS_OLLAMA,
            "model": OLLAMA_MODEL,
            "n_prompts": len(PROMPTS),
            "iterations": args.iterations,
            "warmup": args.warmup,
        },
        "sequential": {
            "mean_s": seq_mean,
            "std_s": seq_std,
            "samples": sequential_times,
        },
        "parallel": {
            "mean_s": par_mean,
            "std_s": par_std,
            "samples": parallel_times,
        },
        "efficiency": {
            "mean_pct": eff_mean,
            "std_pct": eff_std,
            "samples": efficiencies,
        },
        "speedup": {
            "actual": speedup,
            "theoretical": theoretical_speedup,
            "ratio": speedup / theoretical_speedup if theoretical_speedup > 0 else 0,
        },
        "claim_verification": {
            "paper_claims": "~85% efficiency",
            "measured": f"{eff_mean:.1f}%",
            "verified": 70 <= eff_mean <= 100,
        }
    }

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Sequential ({len(PROMPTS)} ops):  {seq_mean:.2f}s +/- {seq_std:.2f}s")
    print(f"Parallel   ({len(PROMPTS)} ops):  {par_mean:.2f}s +/- {par_std:.2f}s")
    print(f"Speedup:                 {speedup:.2f}x (theoretical: {theoretical_speedup}x)")
    print(f"Efficiency:              {eff_mean:.1f}% +/- {eff_std:.1f}%")
    print()
    print("=" * 60)
    print("PAPER CLAIM VERIFICATION")
    print("=" * 60)
    print(f"Paper claims: ~85% efficiency")
    print(f"Measured:     {eff_mean:.1f}%")
    if results["claim_verification"]["verified"]:
        print("CLAIM: VERIFIED")
    else:
        print("CLAIM: NEEDS UPDATE")
    print()

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        # Output JSON
        print("\n--- JSON OUTPUT ---")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
