#!/usr/bin/env python3
"""
Experiment 1: Speedup vs Fusion Count

Measures latency for 1, 2, 3, 5, 8 RSN operations (fused vs unfused).
Expected: ~N× speedup for N operations (I/O bound).

This validates the claim that FuseReasoning provides N× speedup for N operations.
"""

import asyncio
import time
import json
import statistics
from dataclasses import dataclass
from typing import List, Dict, Any
import subprocess

# Ollama configuration
MODEL = "gpt-oss:120b-cloud"  # Cloud model for accurate benchmarks
OLLAMA_URL = "http://localhost:11434/api/generate"

@dataclass
class ExperimentResult:
    fusion_count: int
    unfused_latency_ms: float
    fused_latency_ms: float
    speedup: float
    unfused_latency_std: float
    fused_latency_std: float

async def call_ollama(prompt: str, timeout: float = 60.0) -> tuple[str, float]:
    """Call Ollama and return response with latency."""
    import aiohttp

    start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            result = await resp.json()
            latency = (time.perf_counter() - start) * 1000  # ms
            return result.get("response", ""), latency

async def run_unfused(prompts: List[str]) -> tuple[List[str], float]:
    """Run prompts sequentially (unfused)."""
    results = []
    total_latency = 0

    for prompt in prompts:
        response, latency = await call_ollama(prompt)
        results.append(response)
        total_latency += latency

    return results, total_latency

async def run_fused(prompts: List[str]) -> tuple[str, float]:
    """Run prompts as a single fused call."""
    # Construct fused prompt with separator (mimics FuseReasoning pass)
    fused_prompt = "\n---\n".join([
        f"Task {i+1}: {prompt}"
        for i, prompt in enumerate(prompts)
    ])
    fused_prompt += "\n---\nProvide answers for all tasks above, numbered 1 through " + str(len(prompts)) + "."

    response, latency = await call_ollama(fused_prompt)
    return response, latency

def generate_classification_prompts(count: int) -> List[str]:
    """Generate simple classification prompts (fusion-friendly)."""
    topics = [
        "machine learning", "quantum computing", "blockchain",
        "neural networks", "cybersecurity", "cloud computing",
        "data science", "artificial intelligence"
    ]
    return [
        f"In one sentence, define '{topics[i % len(topics)]}'."
        for i in range(count)
    ]

async def run_experiment(fusion_count: int, trials: int = 5) -> ExperimentResult:
    """Run experiment for a specific fusion count."""
    prompts = generate_classification_prompts(fusion_count)

    unfused_latencies = []
    fused_latencies = []

    for trial in range(trials):
        print(f"  Trial {trial + 1}/{trials} for {fusion_count} ops...", end=" ", flush=True)

        # Run unfused
        _, unfused_lat = await run_unfused(prompts)
        unfused_latencies.append(unfused_lat)

        # Run fused
        _, fused_lat = await run_fused(prompts)
        fused_latencies.append(fused_lat)

        print(f"unfused={unfused_lat:.0f}ms, fused={fused_lat:.0f}ms")

    unfused_mean = statistics.mean(unfused_latencies)
    fused_mean = statistics.mean(fused_latencies)

    return ExperimentResult(
        fusion_count=fusion_count,
        unfused_latency_ms=unfused_mean,
        fused_latency_ms=fused_mean,
        speedup=unfused_mean / fused_mean if fused_mean > 0 else 0,
        unfused_latency_std=statistics.stdev(unfused_latencies) if len(unfused_latencies) > 1 else 0,
        fused_latency_std=statistics.stdev(fused_latencies) if len(fused_latencies) > 1 else 0,
    )

async def main():
    print("=" * 60)
    print("Experiment 1: Speedup vs Fusion Count")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print()

    # Check Ollama is running
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if MODEL.split(":")[0] not in result.stdout:
            print(f"Warning: Model {MODEL} may not be available. Run: ollama pull {MODEL}")
    except Exception as e:
        print(f"Warning: Could not check Ollama status: {e}")

    fusion_counts = [1, 2, 3, 5, 8]
    results: List[ExperimentResult] = []

    for count in fusion_counts:
        print(f"\nRunning with {count} operations:")
        result = await run_experiment(count, trials=3)
        results.append(result)

    # Print results table
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"{'Ops':>4} | {'Unfused (ms)':>14} | {'Fused (ms)':>14} | {'Speedup':>8} | {'Expected':>8}")
    print("-" * 80)

    for r in results:
        expected = r.fusion_count
        print(f"{r.fusion_count:>4} | {r.unfused_latency_ms:>10.0f} +/- {r.unfused_latency_std:>4.0f} | "
              f"{r.fused_latency_ms:>10.0f} +/- {r.fused_latency_std:>4.0f} | "
              f"{r.speedup:>7.2f}x | {expected:>7.1f}x")

    # Export JSON
    output = {
        "experiment": "speedup_vs_fusion_count",
        "model": MODEL,
        "results": [
            {
                "fusion_count": r.fusion_count,
                "unfused_latency_ms": r.unfused_latency_ms,
                "fused_latency_ms": r.fused_latency_ms,
                "speedup": r.speedup,
                "expected_speedup": r.fusion_count,
                "speedup_efficiency": r.speedup / r.fusion_count if r.fusion_count > 0 else 0,
            }
            for r in results
        ]
    }

    with open("speedup_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to speedup_results.json")

    # Summary for paper
    print("\n" + "=" * 60)
    print("FOR PAPER (tab/fuse-speedup.tex):")
    print("=" * 60)
    print("\\begin{tabular}{rrrr}")
    print("\\toprule")
    print("\\textbf{Ops} & \\textbf{Unfused} & \\textbf{Fused} & \\textbf{Speedup} \\\\")
    print("\\midrule")
    for r in results:
        print(f"{r.fusion_count} & {r.unfused_latency_ms:.0f}ms & {r.fused_latency_ms:.0f}ms & {r.speedup:.2f}$\\times$ \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

if __name__ == "__main__":
    asyncio.run(main())
