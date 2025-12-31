# Workload 4: Scalability

## Purpose

Validate that A-PXM's Work-Span model correctly predicts speedup as parallelism degree increases. This tests pure fan-out parallelism with N independent operations.

## What We're Demonstrating

**A-PXM Property**: Dataflow semantics enable Work-Span model validation

With N independent operations and a critical path of 1 (all operations can run in parallel), theoretical speedup equals N. This workload measures how close A-PXM's scheduler gets to theoretical limits.

```
Work-Span Model:
T_1   = Total work (N operations)
T_inf = Span (critical path = 1)
Speedup_theoretical = T_1 / T_inf = N / 1 = N
```

### A-PXM Code (workflow.ais)

```
agent ScalabilityTest {
    // 2-way parallel
    @entry flow parallel_2() -> str {
        rsn("Task A") -> a
        rsn("Task B") -> b
        merge(a, b) -> result
        return result
    }

    // 4-way parallel
    flow parallel_4() -> str {
        rsn("Task A") -> a
        rsn("Task B") -> b
        rsn("Task C") -> c
        rsn("Task D") -> d
        merge(a, b, c, d) -> result
        return result
    }

    // 8-way parallel
    flow parallel_8() -> str {
        rsn("Task A") -> a
        rsn("Task B") -> b
        rsn("Task C") -> c
        rsn("Task D") -> d
        rsn("Task E") -> e
        rsn("Task F") -> f
        rsn("Task G") -> g
        rsn("Task H") -> h
        merge(a, b, c, d, e, f, g, h) -> result
        return result
    }
}
```

### Expected Speedup Curve

```
Speedup
   ^
 8 |                              / Theoretical (y=x)
   |                            /
 6 |                          /   * A-PXM
   |                        /   *
 4 |                      /   *
   |                    /   *
 2 |                  /   *
   |                /   *
 1 +-------------- *-------------------> N
   1        2        4        8
```

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
cd papers/CF26/benchmarks/workloads/4_scalability

# Run 2-way parallel
apxm compiler run workflow.ais -O1

# Run 4-way and 8-way (requires separate workflow files or runtime selection)
apxm compiler run workflow_n4.ais -O1
apxm compiler run workflow_n8.ais -O1
```

### Run LangGraph Comparison

```bash
cd papers/CF26/benchmarks/workloads/4_scalability
python workflow.py
```

### Run Full Benchmark (Both)

```bash
# From repo root
apxm workloads run 4_scalability

# With JSON output
apxm workloads run 4_scalability --json
```

---

## Results

*To be filled after benchmark execution*

| N | T_1 (Work) | T_inf (Span) | Theoretical | A-PXM Measured | Efficiency |
|---|------------|--------------|-------------|----------------|------------|
| 2 | 2 | 1 | 2x | - | - |
| 4 | 4 | 1 | 4x | - | - |
| 8 | 8 | 1 | 8x | - | - |

---

## Analysis

*To be filled after benchmark execution*

### Expected Observations

1. **Speedup scales with N**: As parallelism degree increases, speedup should approach theoretical limits.

2. **Bounded by I/O variance**: Real speedup may be limited by LLM response time variance rather than scheduling overhead.

3. **Scheduler overhead negligible**: A-PXM's ~7.5us per-operation overhead is orders of magnitude below LLM latency.

### Key Insight

This workload validates that A-PXM's formal execution model (Work-Span) correctly predicts parallelism behavior. The speedup is a **consequence** of dataflow semantics, not a manually-engineered feature.
