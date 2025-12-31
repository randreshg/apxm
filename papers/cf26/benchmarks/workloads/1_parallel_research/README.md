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
 [RSN 1] [RSN 2] [RSN 3]   <- PARALLEL (no dependencies)
    |        |        |
    +--------+--------+
             |
         [MERGE]
             |
         [RSN 4]           <- SEQUENTIAL (depends on merge)
             |
        OUTPUT: report
```

### A-PXM Code (workflow.ais)

```
agent ParallelResearch {
    @entry flow main(topic: str) -> str {
        // These 3 RSN ops have no dependencies -> run in PARALLEL automatically
        rsn("Explain the domain background of " + topic) -> background
        rsn("What are the recent advances in " + topic) -> advances
        rsn("What is the societal impact of " + topic) -> impact

        // MERGE waits for all 3, then synthesize
        merge(background, advances, impact) -> combined

        // Final synthesis is sequential (depends on merge)
        rsn("Synthesize into a coherent report: " + combined) -> report
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

### Prerequisites

```bash
# Start Ollama (local LLM backend)
ollama serve
ollama pull gpt-oss:20b-cloud

# Install Python dependencies
pip install langgraph langchain-ollama

# Build A-PXM compiler (from repo root)
apxm compiler build
```

### Run A-PXM Version

```bash
cd papers/CF26/benchmarks/workloads/1_parallel_research

# Compile the workflow
apxm compiler compile workflow.ais -o workflow

# Run with execution (includes compilation)
apxm compiler run workflow.ais -O1
```

### Run LangGraph Comparison

```bash
cd papers/CF26/benchmarks/workloads/1_parallel_research
python workflow.py
```

### Run Full Benchmark (Both)

```bash
# From repo root
apxm workloads run 1_parallel_research

# With JSON output
apxm workloads run 1_parallel_research --json
```

---

## Results

*To be filled after benchmark execution*

| Metric | LangGraph | A-PXM | Notes |
|--------|-----------|-------|-------|
| Mean latency (ms) | - | - | |
| LLM Calls | 4 | 4 | Same work |
| Critical Path | 4 | 2 | Structural property |
| Theoretical Speedup | 1x | 2x | T_1 / T_inf |
| Measured Speedup | - | - | |
| Lines of Code | ~45 | ~12 | |

---

## Analysis

*To be filled after benchmark execution*

### Expected Observations

1. **Parallelism emerges from dataflow**: The three RSN operations have no data dependencies, so A-PXM's scheduler fires them concurrently without explicit coordination.

2. **Work-Span model validation**: With T_1 = 4 LLM calls and T_inf = 2 (critical path), theoretical speedup is 2x.

3. **Framework overhead negligible**: A-PXM's scheduling overhead (~7.5us per operation) is orders of magnitude below LLM latency.

### Key Insight

This workload demonstrates that parallelism is a **consequence** of A-PXM's formal execution semantics, not a feature that needs manual implementation. Any framework could achieve similar parallelism with manual coordination - the value is that A-PXM extracts it automatically from the program structure.
