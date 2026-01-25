# Workload 1: Parallel Research

## Purpose

Demonstrate that A-PXM's dataflow semantics enable automatic parallelism. Three independent research operations execute concurrently without manual async/await coordination.

## What We're Demonstrating

**A-PXM Property**: Dataflow execution semantics

When operations have no data dependencies, A-PXM's scheduler fires them in parallel automatically. The developer writes sequential-looking code; parallelism emerges from the program structure.

```
        INPUT: topic
             |
    +--------+--------+
    |        |        |
    v        v        v
 [ASK 1] [ASK 2] [ASK 3]   <- PARALLEL (no dependencies)
    |        |        |
    +--------+--------+
             |
         [MERGE]
             |
         [ASK 4]           <- SEQUENTIAL (depends on merge)
             |
        OUTPUT: report
```

### A-PXM Code (workflow.ais)

```
agent ParallelResearch {
    @entry flow main(topic: str) -> str {
        // These 3 ask ops have no dependencies -> run in PARALLEL automatically
        ask("Explain the domain background of " + topic) -> background
        ask("What are the recent advances in " + topic) -> advances
        ask("What is the societal impact of " + topic) -> impact

        // MERGE waits for all 3, then synthesize
        merge(background, advances, impact) -> combined

        // Final synthesis is sequential (depends on merge)
        ask("Synthesize into a coherent report: " + combined) -> report
        return report
    }
}
```

### LangGraph Comparison

LangGraph requires explicit `Send` API for parallel execution:
- Manual fan-out with `Send("research_background", state)`
- Explicit edge definitions for synchronization
- ~45 lines vs ~12 lines for A-PXM

---

## How to Run

### Quick Run (Compile + Execute)

```bash
cd papers/cf26/benchmarks/workloads/1_parallel_research

# Compile and execute in one step
apxm execute workflow.ais "quantum computing"

# With tracing to see scheduler behavior (options before file)
apxm execute --trace info workflow.ais "quantum computing"

# With metrics export
apxm execute --emit-metrics metrics.json workflow.ais "quantum computing"
```

### Compile Only

```bash
# Compile to artifact
apxm compile workflow.ais -o workflow.apxmobj

# Compile with diagnostics export
apxm compile workflow.ais -o workflow.apxmobj --emit-diagnostics diagnostics.json -O1
```

### Run Pre-compiled Artifact

```bash
# Run a pre-compiled artifact
apxm run workflow.apxmobj "quantum computing"

# Run with metrics export (options must come BEFORE the file)
apxm run --emit-metrics metrics.json workflow.apxmobj "quantum computing"
```

### Run LangGraph Comparison

```bash
python workflow.py
```

### Run Full Benchmark

```bash
# From repo root
apxm workloads run 1_parallel_research

# With JSON output
apxm workloads run 1_parallel_research --json
```

---

## Collecting Metrics

### Compiler Diagnostics (`--emit-diagnostics`)

Export DAG structure and compilation statistics:

```bash
apxm compile workflow.ais -o workflow.apxmobj --emit-diagnostics diagnostics.json -O1
```

Output includes:
- `dag_statistics`: total_nodes, entry_nodes, exit_nodes, total_edges
- `compilation_phases`: total_ms, artifact_gen_ms
- `passes_applied`: list of optimization passes applied

### Runtime Metrics (`--emit-metrics`)

Export execution performance data (options must come before the file):

```bash
apxm execute --emit-metrics metrics.json workflow.ais "quantum computing"
```

Output includes:
- `execution`: nodes_executed, nodes_failed, duration_ms
- `scheduler`: per_op_overhead_us, max_parallelism, avg_parallelism
- `llm`: total_requests, input/output tokens, latency percentiles

---

## Results

### Measured Values (topic: "machine learning")

| Metric | A-PXM | Notes |
|--------|-------|-------|
| Total Duration | 4300ms | End-to-end execution |
| LLM Calls | 4 | Same as LangGraph |
| Avg LLM Latency | 2088ms | Per-request average |
| Max Parallelism | 3 | Concurrent LLM calls |
| Avg Parallelism | 2.5 | Work-span derived |
| Framework Overhead | ~1.9µs/op | Scheduling cost (input_collection + token_routing) |
| Compilation Time | 70ms | Full pipeline with O1 optimizations |
| DAG Nodes | 23 | IR operations |
| DAG Edges | 22 | Data dependencies |

### Work-Span Analysis

| Metric | Value | Formula |
|--------|-------|---------|
| T_1 (Total Work) | 4 LLM calls | Sequential execution cost |
| T_∞ (Critical Path) | 2 LLM calls | Longest dependency chain |
| Theoretical Speedup | 2x | T_1 / T_∞ |
| Measured Parallelism | 2.5 | Actual concurrent operations |

---

## Analysis

### Observations

1. **Parallelism emerges from dataflow**: The three independent `ask` operations execute concurrently (max_parallelism=3) without any explicit async/await coordination. The developer writes sequential-looking code; the scheduler extracts parallelism automatically.

2. **Work-Span model validated**: With T_1 = 4 LLM calls and T_∞ = 2 (critical path: first parallel batch + synthesis), theoretical speedup is 2x. Measured avg_parallelism=2.5 confirms the model.

3. **Framework overhead negligible**: Scheduling overhead (~1.9µs per operation) is 6 orders of magnitude below LLM latency (~2000ms). A-PXM adds virtually zero overhead to agentic workloads.

4. **Compilation cost amortized**: The 70ms compilation time is a one-time cost. Pre-compiled artifacts (`.apxmobj`) can be reused across executions.

### Key Insight

This workload demonstrates that parallelism is a **consequence** of A-PXM's formal dataflow semantics, not a feature requiring manual implementation. The developer expresses intent; the compiler and runtime extract optimal parallelism from the program structure.
