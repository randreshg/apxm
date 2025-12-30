# A-PXM Benchmark Architecture

This document explains how the benchmark suite collects and captures metrics for the CF'26 paper.

## Benchmark Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BENCHMARK EXECUTION FLOW                             │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │  DSL Source  │
                              │ (workflow.ais)│
                              └──────┬───────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │              STEP 1: COMPILE (once)                    │
        │  ┌────────────────────────────────────────────────┐    │
        │  │  apxm compile workflow.ais -O1                 │    │
        │  │  --emit-diagnostics diagnostics.json           │    │
        │  │  -o workflow.apxmobj                           │    │
        │  └────────────────────────────────────────────────┘    │
        │                                                        │
        │  COLLECT:                                              │
        │  • compilation_phases (total_ms, artifact_gen_ms)      │
        │  • dag_statistics (nodes, edges, entry/exit)           │
        │  • passes_applied (with per-pass statistics)           │
        └────────────────────────────────────────────────────────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │              STEP 2: WARMUP (discard)                  │
        │                                                        │
        │  for i in range(warmup_count):                         │
        │      apxm run workflow.ais -O1  # Output ignored       │
        │                                                        │
        │  Purpose: JIT warmup, cache priming, steady state      │
        │  Result:  DISCARDED - not recorded in output           │
        └────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────┴────────────────────────────────────────┐
│                      STEP 3: MEASUREMENT RUNS                               │
│                                                                             │
│   ┌─────────────────────────────────┐   ┌─────────────────────────────────┐ │
│   │  BUILD A: WITH METRICS          │   │  BUILD B: WITHOUT METRICS       │ │
│   │  (--features driver,metrics)    │   │  (--features driver)            │ │
│   └───────────────┬─────────────────┘   └───────────────┬─────────────────┘ │
│                   │                                     │                   │
│                   ▼                                     ▼                   │
│   ┌─────────────────────────────────┐   ┌─────────────────────────────────┐ │
│   │  for i in range(5):             │   │  for i in range(5):             │ │
│   │    apxm run workflow.ais        │   │    apxm run workflow.ais        │ │
│   │      --emit-metrics out.json    │   │      (no metrics flag)          │ │
│   └───────────────┬─────────────────┘   └───────────────┬─────────────────┘ │
│                   │                                     │                   │
│                   ▼                                     ▼                   │
│   ┌─────────────────────────────────┐   ┌─────────────────────────────────┐ │
│   │  COLLECT (per sample):          │   │  COLLECT (per sample):          │ │
│   │  • wall_time_ms                 │   │  • wall_time_ms                 │ │
│   │  • runtime_only_ms (link_phases)│   │  • runtime_only_ms              │ │
│   │  • scheduler overhead breakdown │   │  • basic execution stats        │ │
│   │  • llm metrics (tokens, latency)│   │                                 │ │
│   │  • execution status             │   │                                 │ │
│   │  • stdout output                │   │                                 │ │
│   └───────────────┬─────────────────┘   └───────────────┬─────────────────┘ │
│                   │                                     │                   │
│                   └──────────────┬──────────────────────┘                   │
│                                  │                                          │
└──────────────────────────────────┼──────────────────────────────────────────┘
                                   │
                                   ▼
        ┌────────────────────────────────────────────────────────┐
        │              STEP 4: AGGREGATE & OUTPUT                │
        │                                                        │
        │  {                                                     │
        │    "input": { "workflow_source": "...", ... },         │
        │    "compiler": { "diagnostics": {...} },               │
        │    "samples_with_metrics": [ {...}, {...}, ... ],      │
        │    "samples_no_metrics": [ {...}, {...}, ... ],        │
        │    "summary": {                                        │
        │      "with_metrics": { "mean_ms": ..., "std_ms": ... },│
        │      "no_metrics": { "mean_ms": ..., "std_ms": ... },  │
        │      "metrics_overhead_pct": X.XX                      │
        │    }                                                   │
        │  }                                                     │
        └────────────────────────────────────────────────────────┘
```

## Build Requirements

Two CLI binaries are needed to measure metrics overhead.

**Using Python CLI (recommended):**

```bash
# Build the compiler (standard, without metrics)
python tools/apxm_cli.py compiler build
```

**Manual (for metrics comparison):**

```bash
# Build WITH metrics (for detailed instrumentation data)
cargo build --release -p apxm-cli --features driver,metrics
cp target/release/apxm target/release/apxm-metrics

# Build WITHOUT metrics (clean baseline, no atomic overhead)
cargo build --release -p apxm-cli --features driver
cp target/release/apxm target/release/apxm-clean
```

## Scheduler Overhead Breakdown

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         SCHEDULER OVERHEAD BREAKDOWN                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────────┐                                                   │
│  │ ready_set_update_us │ Time updating which operations are ready to run  │
│  └─────────────────────┘                                                   │
│             │                                                              │
│             ▼                                                              │
│  ┌─────────────────────┐                                                   │
│  │ work_stealing_us    │ Time workers spend looking for work from others  │
│  └─────────────────────┘                                                   │
│             │                                                              │
│             ▼                                                              │
│  ┌─────────────────────┐                                                   │
│  │ input_collection_us │ Time collecting inputs for an operation          │
│  └─────────────────────┘                                                   │
│             │                                                              │
│             ▼                                                              │
│  ┌─────────────────────┐                                                   │
│  │ operation_dispatch  │ Time dispatching operation to worker             │
│  └─────────────────────┘                                                   │
│             │                                                              │
│             ▼                                                              │
│  ┌─────────────────────┐                                                   │
│  │ token_routing_us    │ Time routing output tokens to consumers          │
│  └─────────────────────┘                                                   │
│                                                                            │
│  per_op_overhead_us = SUM of above / operations_executed                  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

## Link Phases Timeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            LINK PHASES TIMELINE                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  DSL Source                                                                │
│      │                                                                     │
│      ▼ compile_ms (0.35)                                                   │
│  ┌────────┐                                                                │
│  │ Parse  │ → MLIR → Optimize → Lower                                     │
│  └────────┘                                                                │
│      │                                                                     │
│      ▼ artifact_ms (0.03)                                                  │
│  ┌────────────────┐                                                        │
│  │ Generate bytes │ Serialize ExecutionDag to artifact                    │
│  └────────────────┘                                                        │
│      │                                                                     │
│      ▼ validation_ms (0.003)                                               │
│  ┌────────────────┐                                                        │
│  │ Validate DAG   │ Check for cycles, malformed edges                     │
│  └────────────────┘                                                        │
│      │                                                                     │
│      ▼ runtime_ms (780.67)                                                 │
│  ┌────────────────┐                                                        │
│  │ Execute        │ Scheduler + LLM calls + handlers                      │
│  └────────────────┘                                                        │
│                                                                            │
│  KEY: Use runtime_ms to isolate runtime performance from compilation      │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

## Metrics Collection Sources

### 1. Compiler Diagnostics (`--emit-diagnostics`)

Collected once per benchmark run during the compile step.

**Current output:**

```json
{
  "input": "/path/to/workflow.ais",
  "optimization_level": "O1",
  "compilation_phases": {
    "total_ms": 9.87,
    "artifact_gen_ms": 0.19
  },
  "dag_statistics": {
    "total_nodes": 5,
    "entry_nodes": 3,
    "exit_nodes": 2,
    "total_edges": 4
  },
  "passes_applied": [
    "normalize",
    "scheduling",
    "fuse-reasoning",
    "canonicalizer",
    "cse",
    "symbol-dce",
    "lower-to-async"
  ]
}
```

**Enhanced output (target):**

Per-pass statistics will be added to the `--emit-diagnostics` JSON:

```json
{
  "passes": [
    {
      "name": "normalize",
      "timing_ms": 0.82,
      "statistics": {
        "graph_normalized": 5,
        "context_dedups": 3
      }
    },
    {
      "name": "fuse-reasoning",
      "timing_ms": 0.95,
      "statistics": {
        "scanned": 5,
        "fused_pairs": 2
      },
      "token_savings": {
        "estimated_tokens_saved": 500,
        "llm_calls_saved": 2
      }
    }
  ],
  "token_estimation": {
    "method": "tiktoken (cl100k_base)",
    "original_tokens": 1250,
    "optimized_tokens": 750,
    "savings_pct": 40.0
  }
}
```

### 2. Runtime Metrics (`--emit-metrics`)

Collected per iteration during measurement runs.

```json
{
  "input": "/path/to/workflow.ais",
  "optimization_level": "O1",
  "execution": {
    "nodes_executed": 3,
    "nodes_failed": 0,
    "duration_ms": 780,
    "status": "success"
  },
  "scheduler": {
    "per_op_overhead_us": 4.73,
    "overhead_breakdown": {
      "ready_set_update_us": 0.0,
      "work_stealing_us": 3.71,
      "input_collection_us": 0.49,
      "operation_dispatch_us": 0.0,
      "token_routing_us": 0.54
    },
    "max_parallelism": 2,
    "avg_parallelism": 2.0,
    "operations_executed": 3,
    "operations_failed": 0
  },
  "llm": {
    "total_requests": 1,
    "total_input_tokens": 365,
    "total_output_tokens": 38,
    "avg_latency_ms": 778,
    "p50_latency_ms": 778,
    "p99_latency_ms": 778
  },
  "link_phases": {
    "compile_ms": 0.35,
    "artifact_ms": 0.03,
    "validation_ms": 0.003,
    "runtime_ms": 780.67
  }
}
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Compile once, run many | Isolate runtime variance from compilation variance |
| Discard warmup | JIT/cache warmup shouldn't affect statistics |
| Two builds | Quantify metrics instrumentation overhead |
| 5+5 iterations | Sufficient samples for mean/std with/without metrics |
| Extract `runtime_only_ms` | `link_phases.runtime_ms` excludes compile time |
| Full metrics per sample | Scheduler breakdown, LLM latency for analysis |

## Key Metrics Explained

### Metrics Overhead

```
metrics_overhead_pct = (mean_with_metrics - mean_no_metrics) / mean_no_metrics × 100
```

This quantifies the cost of instrumentation (atomic operations on hot path).

### Runtime Isolation

```
runtime_only_ms = link_phases.runtime_ms
```

Use this for runtime benchmarks to exclude compilation time variance.

### Scheduler Efficiency

```
overhead_ratio_pct = (per_op_overhead_us × ops) / (llm_latency_ms × 1000) × 100
```

Target: < 1% overhead relative to LLM latency.

## Output JSON Schema

See [JSON_SCHEMA.md](JSON_SCHEMA.md) for the complete benchmark output format.

## References

- [Compiler Diagnostics Format](COMPILER_DIAGNOSTICS.md)
- [Runtime Metrics Format](RUNTIME_METRICS.md)
- [Testing Plan](../../TESTING_PLAN.md)
