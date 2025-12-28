#!/usr/bin/env python3
"""
LangGraph Baseline Benchmark for A-PXM Paper
=============================================

Fair comparison following "AI Agents That Matter" (arXiv:2407.01502) methodology.

Benchmarks:
1. Pure Overhead: No LLM calls, measure dispatch time only
2. Parallel Efficiency: Compare sequential vs parallel execution
3. State Serialization: Measure checkpoint overhead

Run with:
    pip install -r requirements.txt
    python langgraph_baseline.py

Author: A-PXM Paper Team
Target: Computing Frontiers 2026
"""

import asyncio
import statistics
import time
from typing import TypedDict, Annotated, List
from dataclasses import dataclass

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# For parallel execution
from langgraph.constants import Send

# Optional: For checkpointing benchmarks
try:
    from langgraph.checkpoint.memory import MemorySaver
    HAS_CHECKPOINT = True
except ImportError:
    HAS_CHECKPOINT = False

# Constants matching A-PXM benchmarks
WARMUP_ITERATIONS = 3
BENCHMARK_ITERATIONS = 100
SYNTHETIC_OPS = 500


@dataclass
class BenchmarkResult:
    """Holds benchmark results with statistical measures."""
    name: str
    samples: List[float]
    unit: str = "μs"

    @property
    def mean(self) -> float:
        return statistics.mean(self.samples)

    @property
    def std_dev(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0

    @property
    def min(self) -> float:
        return min(self.samples)

    @property
    def max(self) -> float:
        return max(self.samples)

    @property
    def p50(self) -> float:
        return statistics.median(self.samples)

    @property
    def p99(self) -> float:
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def ci_95(self) -> float:
        """95% confidence interval half-width."""
        import math
        return 1.96 * self.std_dev / math.sqrt(len(self.samples))

    def report(self) -> str:
        return (
            f"  {self.name}:\n"
            f"    Mean:     {self.mean:>10.2f} {self.unit} ± {self.ci_95:.2f} (95% CI)\n"
            f"    Std Dev:  {self.std_dev:>10.2f} {self.unit}\n"
            f"    Min:      {self.min:>10.2f} {self.unit}\n"
            f"    Max:      {self.max:>10.2f} {self.unit}\n"
            f"    P50:      {self.p50:>10.2f} {self.unit}\n"
            f"    P99:      {self.p99:>10.2f} {self.unit}\n"
        )


# =============================================================================
# BENCHMARK 1: Pure Scheduler Overhead (No LLM)
# =============================================================================

class SimpleState(TypedDict):
    """Minimal state for overhead measurement."""
    value: int


def create_noop_node(node_id: int):
    """Create a no-op node that just passes through."""
    def node_fn(state: SimpleState) -> SimpleState:
        return {"value": state["value"] + 1}
    return node_fn


def benchmark_pure_overhead(num_nodes: int = SYNTHETIC_OPS) -> BenchmarkResult:
    """
    Measure pure LangGraph scheduler overhead with no I/O.

    Creates a linear chain of N nodes, each doing minimal work.
    Measures time per node execution.
    """
    print(f"\n{'='*60}")
    print("BENCHMARK 1: Pure Scheduler Overhead")
    print(f"Nodes: {num_nodes}, Iterations: {BENCHMARK_ITERATIONS}")
    print(f"{'='*60}")

    # Build the graph
    builder = StateGraph(SimpleState)

    # Add nodes in a chain
    for i in range(num_nodes):
        builder.add_node(f"node_{i}", create_noop_node(i))

    # Connect nodes: START -> node_0 -> node_1 -> ... -> node_N -> END
    builder.add_edge(START, "node_0")
    for i in range(num_nodes - 1):
        builder.add_edge(f"node_{i}", f"node_{i+1}")
    builder.add_edge(f"node_{num_nodes-1}", END)

    # Compile the graph (without checkpointing for fair comparison)
    graph = builder.compile()

    # Warmup
    print(f"Warming up ({WARMUP_ITERATIONS} iterations)...", end=" ", flush=True)
    for _ in range(WARMUP_ITERATIONS):
        graph.invoke({"value": 0})
    print("done")

    # Benchmark
    samples = []
    print(f"Benchmarking ({BENCHMARK_ITERATIONS} iterations)...", end=" ", flush=True)

    for _ in range(BENCHMARK_ITERATIONS):
        start = time.perf_counter()
        graph.invoke({"value": 0})
        elapsed = time.perf_counter() - start

        # Convert to microseconds per operation
        per_op_us = (elapsed * 1_000_000) / num_nodes
        samples.append(per_op_us)

    print("done")

    result = BenchmarkResult("Per-Op Overhead", samples, "μs")
    print(result.report())

    return result


# =============================================================================
# BENCHMARK 2: Parallel Efficiency
# =============================================================================

class ParallelState(TypedDict):
    """State for parallel benchmark."""
    results: Annotated[List[str], add_messages]
    final: str


def simulated_llm_call(delay_ms: float = 100):
    """Simulate an LLM call with a fixed delay."""
    def node_fn(state: dict) -> dict:
        # Simulate LLM latency
        time.sleep(delay_ms / 1000)
        return {"results": [f"result_{time.time()}"]}
    return node_fn


def benchmark_parallel_efficiency(
    num_parallel: int = 2,
    simulated_delay_ms: float = 100
) -> tuple[BenchmarkResult, BenchmarkResult, float]:
    """
    Compare sequential vs parallel execution.

    Creates a graph with N parallel branches, each simulating an LLM call.
    Measures speedup achieved by parallel execution.
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARK 2: Parallel Efficiency ({num_parallel}-way)")
    print(f"Simulated LLM delay: {simulated_delay_ms}ms, Iterations: {BENCHMARK_ITERATIONS}")
    print(f"{'='*60}")

    # --- Sequential Graph ---
    seq_builder = StateGraph(ParallelState)

    for i in range(num_parallel):
        seq_builder.add_node(f"llm_{i}", simulated_llm_call(simulated_delay_ms))

    seq_builder.add_node("merge", lambda s: {"final": "merged"})

    # Sequential: START -> llm_0 -> llm_1 -> ... -> merge -> END
    seq_builder.add_edge(START, "llm_0")
    for i in range(num_parallel - 1):
        seq_builder.add_edge(f"llm_{i}", f"llm_{i+1}")
    seq_builder.add_edge(f"llm_{num_parallel-1}", "merge")
    seq_builder.add_edge("merge", END)

    seq_graph = seq_builder.compile()

    # --- Parallel Graph (using Send API) ---
    par_builder = StateGraph(ParallelState)

    # Fan-out node
    def fan_out(state: ParallelState):
        return [Send(f"llm_{i}", state) for i in range(num_parallel)]

    for i in range(num_parallel):
        par_builder.add_node(f"llm_{i}", simulated_llm_call(simulated_delay_ms))

    par_builder.add_node("merge", lambda s: {"final": "merged"})

    # Parallel: START -> fan_out -> [llm_0, llm_1, ...] -> merge -> END
    par_builder.add_conditional_edges(START, fan_out)
    for i in range(num_parallel):
        par_builder.add_edge(f"llm_{i}", "merge")
    par_builder.add_edge("merge", END)

    par_graph = par_builder.compile()

    # Warmup
    print(f"Warming up ({WARMUP_ITERATIONS} iterations each)...", end=" ", flush=True)
    for _ in range(WARMUP_ITERATIONS):
        seq_graph.invoke({"results": [], "final": ""})
        par_graph.invoke({"results": [], "final": ""})
    print("done")

    # Benchmark Sequential
    seq_samples = []
    print(f"Benchmarking sequential ({BENCHMARK_ITERATIONS} iterations)...", end=" ", flush=True)

    for _ in range(BENCHMARK_ITERATIONS):
        start = time.perf_counter()
        seq_graph.invoke({"results": [], "final": ""})
        elapsed = time.perf_counter() - start
        seq_samples.append(elapsed * 1000)  # Convert to ms

    print("done")

    # Benchmark Parallel
    par_samples = []
    print(f"Benchmarking parallel ({BENCHMARK_ITERATIONS} iterations)...", end=" ", flush=True)

    for _ in range(BENCHMARK_ITERATIONS):
        start = time.perf_counter()
        par_graph.invoke({"results": [], "final": ""})
        elapsed = time.perf_counter() - start
        par_samples.append(elapsed * 1000)  # Convert to ms

    print("done")

    seq_result = BenchmarkResult("Sequential", seq_samples, "ms")
    par_result = BenchmarkResult("Parallel", par_samples, "ms")

    # Calculate speedup
    speedup = seq_result.mean / par_result.mean
    theoretical = float(num_parallel)
    efficiency = (speedup / theoretical) * 100

    print(seq_result.report())
    print(par_result.report())
    print(f"  Speedup: {speedup:.2f}x (theoretical: {theoretical:.2f}x)")
    print(f"  Efficiency: {efficiency:.1f}%")

    return seq_result, par_result, speedup


# =============================================================================
# BENCHMARK 3: State Serialization Overhead
# =============================================================================

class LargeState(TypedDict):
    """State with varying size for serialization benchmarks."""
    data: dict
    counter: int


def benchmark_state_serialization(state_sizes: List[int] = [10, 100, 1000]) -> dict:
    """
    Measure state serialization overhead at different state sizes.

    LangGraph serializes state to MsgPack for checkpointing.
    This measures the overhead of larger state objects.
    """
    print(f"\n{'='*60}")
    print("BENCHMARK 3: State Serialization Overhead")
    print(f"State sizes: {state_sizes}, Iterations: {BENCHMARK_ITERATIONS}")
    print(f"{'='*60}")

    if not HAS_CHECKPOINT:
        print("WARNING: Checkpoint not available, skipping serialization benchmark")
        return {}

    results = {}

    for size in state_sizes:
        # Create state with N keys
        initial_state = {
            "data": {f"key_{i}": f"value_{i}" * 10 for i in range(size)},
            "counter": 0
        }

        # Build graph with checkpointing
        builder = StateGraph(LargeState)

        def increment(state: LargeState) -> LargeState:
            return {"counter": state["counter"] + 1}

        builder.add_node("increment", increment)
        builder.add_edge(START, "increment")
        builder.add_edge("increment", END)

        # Compile with checkpointing
        checkpointer = MemorySaver()
        graph = builder.compile(checkpointer=checkpointer)

        # Warmup
        for i in range(WARMUP_ITERATIONS):
            graph.invoke(initial_state, {"configurable": {"thread_id": f"warmup_{i}"}})

        # Benchmark
        samples = []
        for i in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter()
            graph.invoke(initial_state, {"configurable": {"thread_id": f"bench_{i}"}})
            elapsed = time.perf_counter() - start
            samples.append(elapsed * 1000)  # Convert to ms

        result = BenchmarkResult(f"State size {size}", samples, "ms")
        results[size] = result
        print(result.report())

    return results


# =============================================================================
# BENCHMARK 4: Comparison with A-PXM Target
# =============================================================================

def comparison_summary(overhead_result: BenchmarkResult):
    """Generate comparison table with A-PXM."""
    print(f"\n{'='*60}")
    print("COMPARISON: LangGraph vs A-PXM")
    print(f"{'='*60}")

    apxm_overhead_us = 8.38  # From A-PXM benchmark
    langgraph_overhead_us = overhead_result.mean

    # Convert if needed (LangGraph might be in ms)
    if langgraph_overhead_us < 1:  # Likely in ms, convert to μs
        langgraph_overhead_us *= 1000

    speedup = langgraph_overhead_us / apxm_overhead_us

    print(f"""
┌─────────────────────────────────────────────────────────────┐
│                  SCHEDULER OVERHEAD COMPARISON               │
├─────────────────────────────────────────────────────────────┤
│  Framework      │  Per-Op Overhead   │  Relative            │
├─────────────────────────────────────────────────────────────┤
│  A-PXM          │  {apxm_overhead_us:>8.2f} μs       │  1.00x (baseline)    │
│  LangGraph      │  {langgraph_overhead_us:>8.2f} μs       │  {speedup:>5.0f}x slower       │
└─────────────────────────────────────────────────────────────┘

Note: A-PXM achieves lower overhead via:
  - Rust's zero-cost abstractions
  - Compile-time DAG construction (vs runtime)
  - Lock-free DashMap registry (vs GIL-limited dict)
  - Automatic dataflow parallelism (vs manual Send API)
""")

    return speedup


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     LANGGRAPH BASELINE BENCHMARK SUITE                       ║
║     For A-PXM Paper - Computing Frontiers 2026               ║
║                                                              ║
║     Following "AI Agents That Matter" (arXiv:2407.01502)     ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Run all benchmarks
    overhead_result = benchmark_pure_overhead(SYNTHETIC_OPS)

    # Parallel efficiency at 2 and 4 branches
    par_results = {}
    for n in [2, 4]:
        seq, par, speedup = benchmark_parallel_efficiency(n, simulated_delay_ms=100)
        par_results[n] = {"seq": seq, "par": par, "speedup": speedup}

    # State serialization (if checkpoint available)
    if HAS_CHECKPOINT:
        ser_results = benchmark_state_serialization([10, 100, 1000])
    else:
        print("\nSkipping serialization benchmark (checkpoint not available)")
        ser_results = {}

    # Final comparison
    speedup = comparison_summary(overhead_result)

    # Summary table for paper
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    PAPER METRICS SUMMARY                     ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  LANGGRAPH RESULTS:                                          ║
║    Per-Op Overhead:     {overhead_result.mean:>8.2f} ± {overhead_result.ci_95:.2f} μs (95% CI)        ║
║    2-way Parallelism:   {par_results[2]['speedup']:.2f}x speedup                           ║
║    4-way Parallelism:   {par_results[4]['speedup']:.2f}x speedup                           ║
║                                                              ║
║  A-PXM COMPARISON:                                           ║
║    Overhead Ratio:      {speedup:.0f}x slower than A-PXM               ║
║                                                              ║
║  METHODOLOGY:                                                ║
║    Iterations:          {BENCHMARK_ITERATIONS}                                      ║
║    Operations/iter:     {SYNTHETIC_OPS}                                     ║
║    Warmup iterations:   {WARMUP_ITERATIONS}                                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    return {
        "overhead": overhead_result,
        "parallel": par_results,
        "serialization": ser_results,
        "speedup_vs_apxm": speedup
    }


if __name__ == "__main__":
    results = main()
