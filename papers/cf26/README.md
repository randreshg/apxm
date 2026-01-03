# A-PXM Evaluation for CF'26

## What is A-PXM?

A-PXM (Agent Program Execution Model) is a **specification** for agent execution, not a competing framework. It defines the formal semantics that any agent system could adopt.

| Component | What It Specifies |
|-----------|-------------------|
| **AAM** | Agent Abstract Machine - explicit state (Beliefs, Goals, Capabilities) + 3-tier memory |
| **AIS** | Agent Instruction Set - 19 typed operations with formal semantics |
| **Dataflow Semantics** | Token-based scheduling - operations fire when input data arrives |

The Rust compiler and runtime in this repository are a **reference implementation** demonstrating the specification works. Any framework (LangGraph, CrewAI, AutoGen) could adopt A-PXM principles.

---

## Unique Contributions (vs Prior Work)

| Contribution | Why It's Novel |
|--------------|----------------|
| **Formal Execution Model** | First PXM for agentic AI (analogous to LLVM IR for compilers) |
| **Typed IR (AIS)** | Enables static verification + compiler optimizations |
| **Explicit 3-Tier Memory** | STM/LTM/Episodic with typed operations (not implicit state) |
| **Dataflow Semantics** | Automatic parallelism from program structure |
| **Full Agent Programs** | Beyond function calling - complete multi-step workflows |

---

## Why Compare to LangGraph?

LangGraph is the most widely-used orchestration framework for agent workflows:
- Production-ready, well-documented
- Represents the "state of practice" for agent development
- Provides a fair baseline for workflow-level comparison

**We are NOT claiming**: "Rust is faster than Python" or "Our runtime beats LangGraph"

**We ARE demonstrating**: What becomes possible when workflows follow A-PXM's formal execution model:
- Typed operations catch errors at compile time
- Explicit state enables inspection and debugging
- Dataflow semantics expose parallelism automatically

---

## How vs LLMCompiler?

LLMCompiler (ICML 2024) introduced DAG-based parallel function calling. A-PXM is complementary:

| Aspect | LLMCompiler | A-PXM |
|--------|-------------|-------|
| **Scope** | Function calls | Full agent programs |
| **State model** | Implicit | Explicit AAM (B,G,C) |
| **IR** | Untyped DAG | Typed AIS (19 operations) |
| **Memory** | None | 3-tier (STM/LTM/Episodic) |
| **Verification** | None | Static type checking |

LLMCompiler could target AIS as an optimization backend - they solve different problems.

---

## Benchmark Workloads

Each workload demonstrates a specific A-PXM property:

| # | Name | A-PXM Property Demonstrated |
|---|------|----------------------------|
| 1 | Parallel Research | Dataflow semantics → automatic parallelism |
| 2 | Chain Fusion | Typed IR → FuseReasoning optimization |
| 3 | Type Verification | Typed operations → compile-time errors |
| 4 | Scalability | Dataflow → Work-Span model validation |
| 5 | Memory Augmented | AAM → 3-tier memory (STM/LTM/Episodic) |
| 6 | Tool Invocation | AIS INV → typed tool calls |
| 7 | Reflection | AIS REFL → native reflection operation |
| 8 | Planning | AIS PLAN → native planning operation |
| 9 | Conditional Routing | Dataflow → parallel preparation |
| 10 | Multi-Agent | AAM → multi-agent coordination |
| 11 | Compilation Scaling | Compiler → linear scaling |
| 12 | Real LLM Probe | Runtime → production feasibility |
| 13 | Fusion Quality | FuseReasoning → task-type analysis |
| 14 | Token Estimation | Typed IR → cost estimation |

See `benchmarks/workloads/` for individual workload documentation.

---

## Running Benchmarks

### Prerequisites

```bash
# 1. Install/update conda environment (includes LangGraph for comparison)
apxm install

# 2. Activate the environment
conda activate apxm

# 3. Start Ollama (local LLM backend)
ollama serve
ollama pull gpt-oss:20b-cloud

# 4. Build A-PXM compiler
apxm build
```

### Quick Start

```bash
# Run all workloads with CSV table generation
apxm benchmarks run --workloads --tables

# Quick mode (3 iterations instead of 10)
apxm benchmarks run --workloads --quick --tables
```

### Run Specific Workloads

```bash
# Run a single workload (e.g., workload 1: Parallel Research)
apxm benchmarks run --workloads --workload 1

# Run multiple workloads
apxm benchmarks run --workloads --workload 1,2,5

# With quick mode and table generation
apxm benchmarks run --workloads --quick --workload 1 --tables
```

### Output Structure

Results are saved to `papers/CF26/benchmarks/results/`:

```
results/
  benchmark_<timestamp>.json           # Combined results (all workloads)
  run_<timestamp>/
    manifest.json                      # Run metadata and artifact index
    tables/summary.csv                 # CSV summary table
    workload_parallel_research.json    # Per-workload detailed results
    workload_chain_fusion.json
    ...
```

### Output Format: CSV

The `summary.csv` provides one row per benchmark for paper tables:

| Column | Description |
|--------|-------------|
| `workload` | Benchmark name (e.g., `parallel_research`) |
| `description` | What the benchmark demonstrates |
| `apxm_mean_ms` | A-PXM mean latency |
| `apxm_std_ms` | Standard deviation |
| `apxm_p50_ms` | Median (50th percentile) |
| `apxm_p95_ms` | 95th percentile |
| `apxm_compile_ms` | Compile time |
| `apxm_llm_ms` | Total LLM call time |
| `apxm_input_tokens` | Mean input tokens |
| `apxm_output_tokens` | Mean output tokens |
| `lg_mean_ms` | LangGraph mean latency |
| `lg_std_ms` | LangGraph standard deviation |
| `lg_p50_ms` | LangGraph median |
| `lg_p95_ms` | LangGraph 95th percentile |
| `speedup` | `lg_mean / apxm_mean` |

### Output Format: JSON

Each `workload_*.json` contains detailed per-iteration data:

```json
{
  "meta": { "run_id": "...", "timestamp": "...", "workload": "parallel_research" },
  "suite_config": { "iterations": 10, "warmup": 3 },
  "result": {
    "results": {
      "langgraph": {
        "mean_ms": 22012.89, "std_ms": 7062.03, "p50_ms": 18561.43, "p95_ms": 28979.35,
        "samples": [30136.90, 18561.43, 17340.36],
        "llm": {
          "total_ms_mean": 28315.99, "calls_mean": 4.0,
          "input_tokens_mean": 658.67, "output_tokens_mean": 3360.0
        }
      },
      "apxm": {
        "mean_ms": 6834.87, "std_ms": 1353.85, "p50_ms": 7442.95, "p95_ms": 7744.55,
        "samples": [5283.59, 7442.95, 7778.06],
        "compiler": {
          "diagnostics": {
            "passes_applied": ["normalize", "scheduling", "fuse-ask-ops", "canonicalizer", "cse"],
            "dag_statistics": { "total_nodes": 23, "total_edges": 22 }
          }
        },
        "metrics": {
          "compile_ms": { "mean_ms": 0.42 },
          "llm_total_ms": { "mean_ms": 10610.67 },
          "llm_requests": { "mean_ms": 4.0 },
          "llm_input_tokens": { "mean_ms": 643.33 },
          "llm_output_tokens": { "mean_ms": 919.0 }
        },
        "sample_details": [
          {
            "iteration": 0, "wall_time_ms": 5283.59,
            "runtime_metrics": {
              "execution": { "duration_ms": 5155, "nodes_executed": 23 },
              "llm": { "avg_latency_ms": 2410, "p50_latency_ms": 3162, "total_input_tokens": 653 },
              "scheduler": {
                "avg_parallelism": 2.5, "max_parallelism": 6,
                "overhead_breakdown": {
                  "input_collection_us": 0.85, "token_routing_us": 1.61, "work_stealing_us": 298130.72
                }
              }
            }
          }
        ]
      }
    }
  }
}
```

**Key sections:**
- `samples` — Per-iteration wall times for statistical analysis
- `compiler.diagnostics` — Optimization passes applied, DAG structure
- `metrics` — Aggregated statistics (mean, std, p50, p95) for all metrics
- `sample_details` — Per-iteration breakdown including scheduler overhead

### Example: Full Benchmark Run

```bash
# Run all 10 workloads in quick mode with tables
apxm benchmarks run --workloads --quick --tables

# Expected output:
# Results saved to: papers/CF26/benchmarks/results/benchmark_20260103_XXXXXX.json
# Tables saved to: .../run_20260103_XXXXXX/tables/summary.csv
```

### CLI Reference

| Option | Description |
|--------|-------------|
| `--workloads` | Run DSL comparison workloads |
| `--quick` | Quick mode (3 iterations, 1 warmup) |
| `--tables` | Auto-generate CSV tables after run |
| `--workload N` | Run specific workload(s) (comma-separated) |
| `--iterations N` | Override iteration count |
| `--warmup N` | Override warmup iterations |
| `--json` | Output JSON only (no progress messages) |
| `--list` | List available workloads |

### Verify Dependencies

```bash
apxm doctor
# Should show:
#   LangGraph: <version>
#   LangChain Ollama: <version>
#   LangChain OpenAI: <version>
```

---

## The Core Insight

> *"A-PXM is a specification; it defines the execution substrate's semantics, not a particular implementation."*

The evaluation demonstrates that the **specification works** - parallelism emerges from dataflow, types catch errors early, and the overhead is negligible. Any framework adopting these principles would see similar benefits.
