# Runtime Metrics Format (`--emit-metrics`)

This document describes the JSON output format for the `apxm run --emit-metrics` flag.

## Usage

```bash
apxm run workflow.ais -O1 --emit-metrics metrics.json
```

## Output Format

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

## Field Descriptions

### `execution`

| Field | Description |
|-------|-------------|
| `nodes_executed` | Number of DAG nodes successfully executed |
| `nodes_failed` | Number of nodes that failed |
| `duration_ms` | Total execution time in milliseconds |
| `status` | Execution result: `success` or `failed` |

### `scheduler`

#### `per_op_overhead_us`

Average scheduler overhead per operation in microseconds.

```
per_op_overhead_us = total_overhead / operations_executed
```

Target: < 10μs per operation.

#### `overhead_breakdown`

| Field | Description |
|-------|-------------|
| `ready_set_update_us` | Time updating which operations are ready to run |
| `work_stealing_us` | Time workers spend looking for work from other queues |
| `input_collection_us` | Time collecting input tokens for an operation |
| `operation_dispatch_us` | Time dispatching an operation to a worker |
| `token_routing_us` | Time routing output tokens to downstream consumers |

#### Parallelism Metrics

| Field | Description |
|-------|-------------|
| `max_parallelism` | Maximum concurrent operations observed |
| `avg_parallelism` | Average concurrent operations during execution |
| `operations_executed` | Total operations run |
| `operations_failed` | Operations that failed |

### `llm`

| Field | Description |
|-------|-------------|
| `total_requests` | Number of LLM API calls made |
| `total_input_tokens` | Sum of input tokens across all requests |
| `total_output_tokens` | Sum of output tokens across all requests |
| `avg_latency_ms` | Mean LLM request latency |
| `p50_latency_ms` | Median LLM request latency |
| `p99_latency_ms` | 99th percentile LLM request latency |

### `link_phases`

Timeline breakdown of the `apxm run` command:

| Field | Description |
|-------|-------------|
| `compile_ms` | Time parsing and optimizing the AIS source |
| `artifact_ms` | Time generating the binary artifact |
| `validation_ms` | Time validating the execution DAG |
| `runtime_ms` | Time executing the workflow (scheduler + LLM) |

**Key insight**: Use `runtime_ms` to isolate runtime performance from compilation time.

## Scheduler Overhead Diagram

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

## Overhead Ratio Calculation

To express scheduler overhead as a percentage of LLM latency:

```
overhead_ratio_pct = (per_op_overhead_us × operations_executed) /
                     (avg_latency_ms × total_requests × 1000) × 100
```

Target: < 1% overhead relative to LLM operations.

## Example Analysis

Given this output:

```json
{
  "scheduler": {
    "per_op_overhead_us": 4.73,
    "operations_executed": 3
  },
  "llm": {
    "avg_latency_ms": 778,
    "total_requests": 1
  }
}
```

Overhead ratio:

```
= (4.73 × 3) / (778 × 1 × 1000) × 100
= 14.19 / 778000 × 100
= 0.0018%
```

This shows scheduler overhead is negligible compared to LLM latency.

## Metrics Collection

Metrics are collected by the `MetricsCollector` in the Rust runtime:

- **Location**: `crates/runtime/apxm-runtime/src/observability/metrics.rs`
- **Feature flag**: Requires `--features metrics` at compile time
- **Overhead**: Atomic operations on hot path (typically < 5% runtime impact)

## Feature Flag Behavior

| Build | Metrics Collection | `--emit-metrics` |
|-------|-------------------|------------------|
| `--features driver` | Disabled | Ignored |
| `--features driver,metrics` | Enabled | Writes JSON |
