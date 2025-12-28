#!/usr/bin/env python3
"""
Experiment 4: Task Type Impact

Compares fusion effectiveness across different task types:
- Classification (should help)
- Extraction (should help)
- Multi-step reasoning (might hurt)
- Creative generation (might hurt)

Reports both speedup AND quality for each task type.
"""

import asyncio
import time
import json
import statistics
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Callable
import subprocess

# Ollama configuration
MODEL = "gpt-oss:120b-cloud"  # Cloud model for accurate benchmarks
OLLAMA_URL = "http://localhost:11434/api/generate"

@dataclass
class TaskResult:
    task_type: str
    num_operations: int
    unfused_latency_ms: float
    fused_latency_ms: float
    speedup: float
    unfused_quality: float  # 0-100
    fused_quality: float    # 0-100
    quality_delta: float
    fusion_recommended: bool

# Task definitions
CLASSIFICATION_TASKS = [
    ("Classify this as positive or negative sentiment: 'I love this product!' Answer: positive or negative", "positive"),
    ("Classify this as positive or negative sentiment: 'This is terrible.' Answer: positive or negative", "negative"),
    ("Classify this as positive or negative sentiment: 'Best purchase ever!' Answer: positive or negative", "positive"),
    ("Is this a question or statement: 'The sky is blue.' Answer: question or statement", "statement"),
    ("Is this a question or statement: 'What time is it?' Answer: question or statement", "question"),
]

EXTRACTION_TASKS = [
    ("Extract the date: 'Meeting on December 25, 2024.' Output just the date.", "december 25"),
    ("Extract the email: 'Contact john@example.com for details.' Output just the email.", "john@example.com"),
    ("Extract the price: 'Total: $99.99' Output just the price.", "99.99"),
    ("Extract the name: 'Dr. Jane Smith will attend.' Output just the full name.", "jane smith"),
    ("Extract the country: 'Shipped from Germany.' Output just the country.", "germany"),
]

REASONING_TASKS = [
    # Multi-step reasoning - each answer depends on previous
    ("Step 1: What is 5 + 3? Just output the number.", "8"),
    ("Step 2: Take the previous result and multiply by 2. Just output the number.", "16"),
    ("Step 3: Take the previous result and subtract 6. Just output the number.", "10"),
    ("Step 4: Take the previous result and divide by 2. Just output the number.", "5"),
    ("Step 5: Is the final result greater than 4? Answer: yes or no", "yes"),
]

CREATIVE_TASKS = [
    # Creative generation - hard to evaluate objectively
    ("Write a one-sentence tagline for a coffee shop.", None),  # No expected answer
    ("Create a name for a tech startup.", None),
    ("Suggest a color palette (3 colors) for a nature app.", None),
    ("Write a haiku about programming.", None),
    ("Invent a fictional planet name.", None),
]

async def call_ollama(prompt: str, timeout: float = 60.0) -> Tuple[str, float]:
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
            latency = (time.perf_counter() - start) * 1000
            return result.get("response", "").lower().strip(), latency

def check_answer(response: str, expected: str) -> bool:
    """Check if response contains expected answer."""
    if expected is None:
        return True  # Creative tasks always "pass"
    return expected.lower() in response.lower()

async def run_unfused(tasks: List[Tuple[str, str]]) -> Tuple[float, float]:
    """Run tasks sequentially, return (total_latency, quality_score)."""
    total_latency = 0
    correct = 0

    context = ""  # For reasoning tasks, accumulate context
    for prompt, expected in tasks:
        # For reasoning tasks, include previous context
        full_prompt = context + prompt if context else prompt
        response, latency = await call_ollama(full_prompt)
        total_latency += latency

        if check_answer(response, expected):
            correct += 1

        # Accumulate context for reasoning chains
        context = f"Previous answer: {response}\n"

    quality = correct / len(tasks) * 100
    return total_latency, quality

async def run_fused(tasks: List[Tuple[str, str]]) -> Tuple[float, float]:
    """Run tasks as fused prompt, return (latency, quality_score)."""
    # Construct fused prompt
    fused_prompt = "Complete all tasks below. Number your answers.\n\n"
    for i, (prompt, _) in enumerate(tasks, 1):
        fused_prompt += f"{i}. {prompt}\n"

    response, latency = await call_ollama(fused_prompt)

    # Check quality
    correct = 0
    for _, expected in tasks:
        if check_answer(response, expected):
            correct += 1

    quality = correct / len(tasks) * 100
    return latency, quality

async def run_experiment(
    tasks: List[Tuple[str, str]],
    task_type: str,
    trials: int = 3
) -> TaskResult:
    """Run experiment for a specific task type."""
    unfused_latencies = []
    fused_latencies = []
    unfused_qualities = []
    fused_qualities = []

    for trial in range(trials):
        print(f"  Trial {trial + 1}/{trials}...", end=" ", flush=True)

        # Run unfused
        unfused_lat, unfused_qual = await run_unfused(tasks)
        unfused_latencies.append(unfused_lat)
        unfused_qualities.append(unfused_qual)

        # Run fused
        fused_lat, fused_qual = await run_fused(tasks)
        fused_latencies.append(fused_lat)
        fused_qualities.append(fused_qual)

        print(f"unfused={unfused_lat:.0f}ms/{unfused_qual:.0f}%, fused={fused_lat:.0f}ms/{fused_qual:.0f}%")

    unfused_lat_mean = statistics.mean(unfused_latencies)
    fused_lat_mean = statistics.mean(fused_latencies)
    unfused_qual_mean = statistics.mean(unfused_qualities)
    fused_qual_mean = statistics.mean(fused_qualities)

    speedup = unfused_lat_mean / fused_lat_mean if fused_lat_mean > 0 else 0
    quality_delta = fused_qual_mean - unfused_qual_mean

    # Fusion is recommended if speedup > 1.5x AND quality loss < 10%
    fusion_recommended = speedup > 1.5 and quality_delta > -10

    return TaskResult(
        task_type=task_type,
        num_operations=len(tasks),
        unfused_latency_ms=unfused_lat_mean,
        fused_latency_ms=fused_lat_mean,
        speedup=speedup,
        unfused_quality=unfused_qual_mean,
        fused_quality=fused_qual_mean,
        quality_delta=quality_delta,
        fusion_recommended=fusion_recommended,
    )

async def main():
    print("=" * 60)
    print("Experiment 4: Task Type Impact")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print()

    task_sets = [
        (CLASSIFICATION_TASKS, "classification"),
        (EXTRACTION_TASKS, "extraction"),
        (REASONING_TASKS, "reasoning"),
        (CREATIVE_TASKS, "creative"),
    ]

    results: List[TaskResult] = []

    for tasks, task_type in task_sets:
        print(f"\n{task_type.upper()} Tasks ({len(tasks)} operations):")
        result = await run_experiment(tasks, task_type, trials=3)
        results.append(result)

    # Print results
    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    print(f"{'Task Type':>15} | {'Speedup':>8} | {'Unfused Q':>10} | {'Fused Q':>10} | {'Q Delta':>8} | {'Recommend':>10}")
    print("-" * 100)

    for r in results:
        recommend = "YES" if r.fusion_recommended else "NO"
        print(f"{r.task_type:>15} | {r.speedup:>7.2f}x | {r.unfused_quality:>9.1f}% | "
              f"{r.fused_quality:>9.1f}% | {r.quality_delta:>+7.1f}% | {recommend:>10}")

    # Export JSON
    output = {
        "experiment": "task_type_impact",
        "model": MODEL,
        "results": [
            {
                "task_type": r.task_type,
                "num_operations": r.num_operations,
                "speedup": r.speedup,
                "unfused_quality_pct": r.unfused_quality,
                "fused_quality_pct": r.fused_quality,
                "quality_delta_pct": r.quality_delta,
                "fusion_recommended": r.fusion_recommended,
            }
            for r in results
        ]
    }

    with open("task_type_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to task_type_results.json")

    # Summary for paper
    print("\n" + "=" * 60)
    print("FOR PAPER (Fusion Applicability Table):")
    print("=" * 60)
    print("\\begin{tabular}{lrrrl}")
    print("\\toprule")
    print("\\textbf{Task Type} & \\textbf{Speedup} & \\textbf{Quality $\\Delta$} & \\textbf{Recommend} \\\\")
    print("\\midrule")
    for r in results:
        recommend = "\\checkmark" if r.fusion_recommended else "$\\times$"
        print(f"{r.task_type.capitalize()} & {r.speedup:.1f}$\\times$ & {r.quality_delta:+.1f}\\% & {recommend} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    for r in results:
        status = "RECOMMENDED" if r.fusion_recommended else "NOT RECOMMENDED"
        reason = ""
        if not r.fusion_recommended:
            if r.speedup <= 1.5:
                reason = " (insufficient speedup)"
            elif r.quality_delta <= -10:
                reason = " (quality degradation)"
        print(f"  {r.task_type}: {status}{reason}")

if __name__ == "__main__":
    asyncio.run(main())
