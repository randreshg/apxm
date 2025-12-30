# Benchmark JSON Output Schema

This document defines the complete JSON output format for benchmark runs.

## Top-Level Structure

```json
{
  "meta": { ... },
  "config": { ... },
  "input": { ... },
  "compiler": { ... },
  "samples_with_metrics": [ ... ],
  "samples_no_metrics": [ ... ],
  "summary": { ... }
}
```

## Field Reference

### `meta` - Benchmark Metadata

```json
{
  "meta": {
    "benchmark": "real_llm_probe",
    "timestamp": "2025-12-29T18:48:10+00:00",
    "version": "2.0"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `benchmark` | string | Benchmark name |
| `timestamp` | string | ISO 8601 timestamp |
| `version` | string | Schema version |

### `config` - Run Configuration

```json
{
  "config": {
    "warmup": 2,
    "iterations_with_metrics": 5,
    "iterations_no_metrics": 5,
    "opt_level": 1
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `warmup` | int | Warmup iterations (discarded) |
| `iterations_with_metrics` | int | Measurement runs with metrics |
| `iterations_no_metrics` | int | Measurement runs without metrics |
| `opt_level` | int | Optimization level (0, 1, or 2) |

### `input` - Workflow Source

```json
{
  "input": {
    "workflow_source": "agent LLMProbe { flow main { rsn \"What is 2+2?\" -> answer } }",
    "workflow_path": "/path/to/workflow.ais"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `workflow_source` | string | Complete AIS source code |
| `workflow_path` | string? | Path to source file (null if inline) |

### `compiler` - Compilation Diagnostics

```json
{
  "compiler": {
    "wall_time_ms": 25.5,
    "diagnostics": {
      "optimization_level": "O1",
      "compilation_phases": {
        "total_ms": 9.9,
        "artifact_gen_ms": 0.2
      },
      "dag_statistics": {
        "total_nodes": 3,
        "entry_nodes": 2,
        "exit_nodes": 2,
        "total_edges": 1
      },
      "passes_applied": ["normalize", "scheduling", "fuse-reasoning", "..."]
    }
  }
}
```

See [COMPILER_DIAGNOSTICS.md](COMPILER_DIAGNOSTICS.md) for full details.

### `samples_with_metrics` - Detailed Measurement Samples

Array of per-iteration data collected with `--features metrics`:

```json
{
  "samples_with_metrics": [
    {
      "iteration": 0,
      "timestamp": "2025-12-29T18:48:14+00:00",
      "success": true,
      "wall_time_ms": 823.5,
      "runtime_only_ms": 780.7,
      "runtime": {
        "execution": {
          "status": "success",
          "duration_ms": 780,
          "nodes_executed": 3,
          "nodes_failed": 0
        },
        "scheduler": {
          "per_op_overhead_us": 4.73,
          "max_parallelism": 2,
          "avg_parallelism": 2.0,
          "overhead_breakdown": {
            "ready_set_update_us": 0.0,
            "work_stealing_us": 3.71,
            "input_collection_us": 0.49,
            "operation_dispatch_us": 0.0,
            "token_routing_us": 0.54
          }
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
          "runtime_ms": 780.7
        }
      },
      "output": {
        "stdout": "Executed 3 nodes in 780 ms\n..."
      }
    }
  ]
}
```

See [RUNTIME_METRICS.md](RUNTIME_METRICS.md) for field details.

### `samples_no_metrics` - Baseline Samples

Array of per-iteration data collected without metrics feature:

```json
{
  "samples_no_metrics": [
    {
      "iteration": 0,
      "timestamp": "2025-12-29T18:48:20+00:00",
      "success": true,
      "wall_time_ms": 815.2,
      "output": {
        "stdout": "Executed 3 nodes in 778 ms\n..."
      }
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `iteration` | int | Iteration index (0-based) |
| `timestamp` | string | ISO 8601 timestamp for this sample |
| `success` | bool | Whether execution succeeded |
| `wall_time_ms` | float | Total wall clock time |
| `output.stdout` | string | Captured stdout |

### `summary` - Aggregated Statistics

```json
{
  "summary": {
    "with_metrics": {
      "count": 5,
      "wall_time": {
        "mean_ms": 859.4,
        "std_ms": 44.5,
        "min_ms": 817.9,
        "max_ms": 900.9
      },
      "runtime_only": {
        "mean_ms": 823.5,
        "std_ms": 61.5
      },
      "llm_latency": {
        "mean_ms": 778,
        "std_ms": 12.3
      },
      "tokens": {
        "input_mean": 365,
        "output_mean": 40.5
      }
    },
    "no_metrics": {
      "count": 5,
      "wall_time": {
        "mean_ms": 815.2,
        "std_ms": 38.2
      }
    },
    "metrics_overhead_pct": 5.4
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `with_metrics.count` | int | Number of successful samples |
| `with_metrics.wall_time` | stats | Wall time statistics |
| `with_metrics.runtime_only` | stats | Runtime-only statistics |
| `with_metrics.llm_latency` | stats | LLM latency statistics |
| `with_metrics.tokens` | object | Token usage means |
| `no_metrics.count` | int | Number of baseline samples |
| `no_metrics.wall_time` | stats | Baseline wall time statistics |
| `metrics_overhead_pct` | float | Metrics collection overhead percentage |

#### Statistics Object

```json
{
  "mean_ms": 823.5,
  "std_ms": 61.5,
  "min_ms": 750.0,
  "max_ms": 900.0
}
```

## Complete Example

```json
{
  "meta": {
    "benchmark": "real_llm_probe",
    "timestamp": "2025-12-29T18:48:10+00:00",
    "version": "2.0"
  },
  "config": {
    "warmup": 2,
    "iterations_with_metrics": 5,
    "iterations_no_metrics": 5,
    "opt_level": 1
  },
  "input": {
    "workflow_source": "agent LLMProbe {\n  flow main {\n    rsn \"What is 2+2?\" -> answer\n  }\n}",
    "workflow_path": null
  },
  "compiler": {
    "wall_time_ms": 25.5,
    "diagnostics": {
      "optimization_level": "O1",
      "compilation_phases": {"total_ms": 9.9, "artifact_gen_ms": 0.2},
      "dag_statistics": {"total_nodes": 3, "entry_nodes": 2, "exit_nodes": 2, "total_edges": 1},
      "passes_applied": ["normalize", "scheduling", "fuse-reasoning", "canonicalizer", "cse", "symbol-dce", "lower-to-async"]
    }
  },
  "samples_with_metrics": [
    {
      "iteration": 0,
      "timestamp": "2025-12-29T18:48:14+00:00",
      "success": true,
      "wall_time_ms": 823.5,
      "runtime_only_ms": 780.7,
      "runtime": {
        "execution": {"status": "success", "duration_ms": 780, "nodes_executed": 3, "nodes_failed": 0},
        "scheduler": {"per_op_overhead_us": 4.73, "max_parallelism": 2, "avg_parallelism": 2.0, "overhead_breakdown": {"ready_set_update_us": 0.0, "work_stealing_us": 3.71, "input_collection_us": 0.49, "operation_dispatch_us": 0.0, "token_routing_us": 0.54}},
        "llm": {"total_requests": 1, "total_input_tokens": 365, "total_output_tokens": 38, "avg_latency_ms": 778, "p50_latency_ms": 778, "p99_latency_ms": 778},
        "link_phases": {"compile_ms": 0.35, "artifact_ms": 0.03, "validation_ms": 0.003, "runtime_ms": 780.7}
      },
      "output": {"stdout": "Executed 3 nodes in 780 ms\n"}
    }
  ],
  "samples_no_metrics": [
    {
      "iteration": 0,
      "timestamp": "2025-12-29T18:48:20+00:00",
      "success": true,
      "wall_time_ms": 815.2,
      "output": {"stdout": "Executed 3 nodes in 778 ms\n"}
    }
  ],
  "summary": {
    "with_metrics": {
      "count": 5,
      "wall_time": {"mean_ms": 859.4, "std_ms": 44.5, "min_ms": 817.9, "max_ms": 900.9},
      "runtime_only": {"mean_ms": 823.5, "std_ms": 61.5},
      "llm_latency": {"mean_ms": 778, "std_ms": 12.3},
      "tokens": {"input_mean": 365, "output_mean": 40.5}
    },
    "no_metrics": {
      "count": 5,
      "wall_time": {"mean_ms": 815.2, "std_ms": 38.2}
    },
    "metrics_overhead_pct": 5.4
  }
}
```

## Version History

| Version | Changes |
|---------|---------|
| 2.0 | Added `samples_with_metrics`, `samples_no_metrics`, warmup discarded, per-iteration data |
| 1.0 | Initial format with aggregated results only |
