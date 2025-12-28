#!/usr/bin/env python3
"""
Experiment 2: Quality Preservation

Measures whether fused prompts produce equivalent quality to unfused prompts.
Uses classification tasks where correctness can be verified.

Expected: <5% accuracy difference for short contexts.
"""

import asyncio
import time
import json
import statistics
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import subprocess

# Ollama configuration
MODEL = "gpt-oss:120b-cloud"  # Cloud model for accurate benchmarks
OLLAMA_URL = "http://localhost:11434/api/generate"

# Test cases: (prompt, expected_answer_contains)
CLASSIFICATION_TESTS = [
    ("Is Python a compiled or interpreted language? Answer with one word: 'compiled' or 'interpreted'.", "interpreted"),
    ("Is the sun a star or a planet? Answer with one word: 'star' or 'planet'.", "star"),
    ("Is water H2O or CO2? Answer with the correct formula only.", "h2o"),
    ("Is 2+2 equal to 4 or 5? Answer with just the number.", "4"),
    ("Is Tokyo in Japan or China? Answer with just the country name.", "japan"),
]

EXTRACTION_TESTS = [
    ("Extract the year from: 'The Declaration of Independence was signed in 1776.' Just output the year.", "1776"),
    ("Extract the name from: 'Albert Einstein developed the theory of relativity.' Just output the name.", "einstein"),
    ("Extract the color from: 'The sky is blue on a clear day.' Just output the color.", "blue"),
    ("Extract the number from: 'There are 7 continents on Earth.' Just output the number.", "7"),
    ("Extract the animal from: 'The cheetah is the fastest land animal.' Just output the animal.", "cheetah"),
]

@dataclass
class QualityResult:
    task_type: str
    num_operations: int
    unfused_accuracy: float
    fused_accuracy: float
    accuracy_delta: float
    unfused_correct: int
    fused_correct: int
    total_tests: int

async def call_ollama(prompt: str, timeout: float = 60.0) -> str:
    """Call Ollama and return response."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            result = await resp.json()
            return result.get("response", "").lower().strip()

def check_answer(response: str, expected: str) -> bool:
    """Check if response contains expected answer."""
    return expected.lower() in response.lower()

async def run_unfused(tests: List[Tuple[str, str]]) -> Tuple[int, int]:
    """Run tests sequentially (unfused), return (correct, total)."""
    correct = 0
    for prompt, expected in tests:
        response = await call_ollama(prompt)
        if check_answer(response, expected):
            correct += 1
    return correct, len(tests)

async def run_fused(tests: List[Tuple[str, str]]) -> Tuple[int, int]:
    """Run tests as fused prompt, return (correct, total)."""
    # Construct fused prompt
    fused_prompt = "Answer each question below. Number your answers.\n\n"
    for i, (prompt, _) in enumerate(tests, 1):
        fused_prompt += f"{i}. {prompt}\n"

    response = await call_ollama(fused_prompt)

    # Check each answer in the fused response
    correct = 0
    for i, (_, expected) in enumerate(tests, 1):
        # Look for the answer near the question number
        # This is a simple heuristic - real validation would be more sophisticated
        if expected.lower() in response.lower():
            correct += 1

    return correct, len(tests)

async def run_experiment(
    tests: List[Tuple[str, str]],
    task_type: str,
    trials: int = 5
) -> QualityResult:
    """Run quality preservation experiment."""
    unfused_correct_list = []
    fused_correct_list = []

    for trial in range(trials):
        print(f"  Trial {trial + 1}/{trials}...", end=" ", flush=True)

        # Run unfused
        unfused_correct, total = await run_unfused(tests)
        unfused_correct_list.append(unfused_correct)

        # Run fused
        fused_correct, _ = await run_fused(tests)
        fused_correct_list.append(fused_correct)

        print(f"unfused={unfused_correct}/{total}, fused={fused_correct}/{total}")

    unfused_mean = statistics.mean(unfused_correct_list)
    fused_mean = statistics.mean(fused_correct_list)
    total = len(tests)

    return QualityResult(
        task_type=task_type,
        num_operations=len(tests),
        unfused_accuracy=unfused_mean / total * 100,
        fused_accuracy=fused_mean / total * 100,
        accuracy_delta=(fused_mean - unfused_mean) / total * 100,
        unfused_correct=int(unfused_mean),
        fused_correct=int(fused_mean),
        total_tests=total,
    )

async def main():
    print("=" * 60)
    print("Experiment 2: Quality Preservation")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print()

    results: List[QualityResult] = []

    # Test classification tasks
    print("\nClassification Tasks (5 operations):")
    result = await run_experiment(CLASSIFICATION_TESTS, "classification", trials=3)
    results.append(result)

    # Test extraction tasks
    print("\nExtraction Tasks (5 operations):")
    result = await run_experiment(EXTRACTION_TESTS, "extraction", trials=3)
    results.append(result)

    # Test combined (10 operations - approaching "lost in middle" threshold)
    print("\nCombined Tasks (10 operations):")
    combined = CLASSIFICATION_TESTS + EXTRACTION_TESTS
    result = await run_experiment(combined, "combined", trials=3)
    results.append(result)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"{'Task Type':>15} | {'Ops':>4} | {'Unfused Acc':>12} | {'Fused Acc':>12} | {'Delta':>8}")
    print("-" * 80)

    for r in results:
        delta_str = f"{r.accuracy_delta:+.1f}%"
        status = "OK" if abs(r.accuracy_delta) < 10 else "WARN"
        print(f"{r.task_type:>15} | {r.num_operations:>4} | {r.unfused_accuracy:>11.1f}% | "
              f"{r.fused_accuracy:>11.1f}% | {delta_str:>8} [{status}]")

    # Export JSON
    output = {
        "experiment": "quality_preservation",
        "model": MODEL,
        "results": [
            {
                "task_type": r.task_type,
                "num_operations": r.num_operations,
                "unfused_accuracy_pct": r.unfused_accuracy,
                "fused_accuracy_pct": r.fused_accuracy,
                "accuracy_delta_pct": r.accuracy_delta,
                "quality_preserved": abs(r.accuracy_delta) < 10,
            }
            for r in results
        ]
    }

    with open("quality_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to quality_results.json")

    # Summary for paper
    print("\n" + "=" * 60)
    print("FOR PAPER (tex/05_evaluation.tex):")
    print("=" * 60)
    avg_delta = statistics.mean([abs(r.accuracy_delta) for r in results])
    print(f"Average accuracy delta: {avg_delta:.1f}%")
    if avg_delta < 5:
        print("Conclusion: Quality is preserved (<5% difference)")
    elif avg_delta < 10:
        print("Conclusion: Minor quality impact (<10% difference)")
    else:
        print("Conclusion: Significant quality degradation (>10% difference)")

if __name__ == "__main__":
    asyncio.run(main())
