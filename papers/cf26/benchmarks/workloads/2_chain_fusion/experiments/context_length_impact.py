#!/usr/bin/env python3
"""
Experiment 3: Context Length Impact

Varies combined context from 1K to 10K tokens and measures quality degradation.
Tests the "lost in the middle" phenomenon for fused prompts.

Expected: degradation after ~3-5K tokens.
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

# Filler text to pad context (approximately 100 tokens per paragraph)
FILLER_PARAGRAPH = """
The development of artificial intelligence has transformed numerous industries
and continues to shape the future of technology. Machine learning algorithms
process vast amounts of data to identify patterns and make predictions. Neural
networks, inspired by the human brain, enable computers to learn from experience.
Deep learning techniques have achieved remarkable results in image recognition,
natural language processing, and autonomous systems.
"""

# Target question that will be embedded at different positions
TARGET_QUESTION = "What is the capital of France? Answer with just the city name."
TARGET_ANSWER = "paris"

@dataclass
class ContextResult:
    context_tokens: int
    position: str  # "start", "middle", "end"
    accuracy: float
    avg_latency_ms: float
    trials: int

def estimate_tokens(text: str) -> int:
    """Rough token count estimate (words * 1.3)."""
    return int(len(text.split()) * 1.3)

def generate_padded_prompt(target_tokens: int, position: str) -> str:
    """Generate a prompt with target question at specified position."""
    # Calculate padding needed
    question_tokens = estimate_tokens(TARGET_QUESTION)
    filler_tokens = estimate_tokens(FILLER_PARAGRAPH)
    num_paragraphs = max(0, (target_tokens - question_tokens) // filler_tokens)

    paragraphs_before = 0
    paragraphs_after = 0

    if position == "start":
        paragraphs_after = num_paragraphs
    elif position == "end":
        paragraphs_before = num_paragraphs
    else:  # middle
        paragraphs_before = num_paragraphs // 2
        paragraphs_after = num_paragraphs - paragraphs_before

    # Construct prompt
    prompt = ""
    for i in range(paragraphs_before):
        prompt += f"Section {i+1}:\n{FILLER_PARAGRAPH}\n\n"

    prompt += f"\nIMPORTANT QUESTION: {TARGET_QUESTION}\n\n"

    for i in range(paragraphs_after):
        prompt += f"Section {paragraphs_before + i + 2}:\n{FILLER_PARAGRAPH}\n\n"

    prompt += "\nNow answer the IMPORTANT QUESTION asked above."

    return prompt

async def call_ollama(prompt: str, timeout: float = 120.0) -> Tuple[str, float]:
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

async def run_experiment(
    target_tokens: int,
    position: str,
    trials: int = 5
) -> ContextResult:
    """Run experiment for a specific context length and position."""
    correct = 0
    latencies = []

    prompt = generate_padded_prompt(target_tokens, position)
    actual_tokens = estimate_tokens(prompt)

    for trial in range(trials):
        print(f"  Trial {trial + 1}/{trials} ({actual_tokens} tokens, {position})...", end=" ", flush=True)

        response, latency = await call_ollama(prompt)
        latencies.append(latency)

        is_correct = TARGET_ANSWER in response
        if is_correct:
            correct += 1

        print(f"{'correct' if is_correct else 'wrong'} ({latency:.0f}ms)")

    return ContextResult(
        context_tokens=actual_tokens,
        position=position,
        accuracy=correct / trials * 100,
        avg_latency_ms=statistics.mean(latencies),
        trials=trials,
    )

async def main():
    print("=" * 60)
    print("Experiment 3: Context Length Impact")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Target question: '{TARGET_QUESTION}'")
    print(f"Expected answer contains: '{TARGET_ANSWER}'")
    print()

    # Test different context lengths
    context_sizes = [500, 1000, 2000, 3000, 5000, 8000]
    positions = ["start", "middle", "end"]

    results: List[ContextResult] = []

    for size in context_sizes:
        print(f"\nContext size: ~{size} tokens")
        for position in positions:
            result = await run_experiment(size, position, trials=3)
            results.append(result)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"{'Tokens':>8} | {'Position':>8} | {'Accuracy':>10} | {'Latency':>12}")
    print("-" * 80)

    for r in results:
        status = "OK" if r.accuracy >= 80 else ("WARN" if r.accuracy >= 50 else "FAIL")
        print(f"{r.context_tokens:>8} | {r.position:>8} | {r.accuracy:>9.1f}% | "
              f"{r.avg_latency_ms:>10.0f}ms [{status}]")

    # Analyze "lost in the middle" effect
    print("\n" + "=" * 60)
    print("'LOST IN THE MIDDLE' ANALYSIS")
    print("=" * 60)

    for size in context_sizes:
        size_results = [r for r in results if abs(r.context_tokens - size) < 200]
        if len(size_results) == 3:
            start_acc = next(r.accuracy for r in size_results if r.position == "start")
            middle_acc = next(r.accuracy for r in size_results if r.position == "middle")
            end_acc = next(r.accuracy for r in size_results if r.position == "end")

            avg_edge = (start_acc + end_acc) / 2
            middle_drop = avg_edge - middle_acc

            print(f"{size:>5} tokens: start={start_acc:.0f}%, middle={middle_acc:.0f}%, end={end_acc:.0f}%")
            print(f"           Middle drop: {middle_drop:+.1f}% vs edges")

    # Export JSON
    output = {
        "experiment": "context_length_impact",
        "model": MODEL,
        "results": [
            {
                "context_tokens": r.context_tokens,
                "position": r.position,
                "accuracy_pct": r.accuracy,
                "avg_latency_ms": r.avg_latency_ms,
            }
            for r in results
        ]
    }

    with open("context_length_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to context_length_results.json")

    # Summary for paper
    print("\n" + "=" * 60)
    print("FOR PAPER:")
    print("=" * 60)

    # Find threshold where middle accuracy drops below 80%
    threshold = None
    for size in context_sizes:
        middle_result = next((r for r in results if r.position == "middle" and abs(r.context_tokens - size) < 200), None)
        if middle_result and middle_result.accuracy < 80 and threshold is None:
            threshold = size

    if threshold:
        print(f"Context length threshold: ~{threshold} tokens")
        print("Fusion should be limited to contexts under this size for quality preservation.")
    else:
        print("No significant quality degradation observed in tested range.")

if __name__ == "__main__":
    asyncio.run(main())
