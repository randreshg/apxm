# Compiler Diagnostics Format (`--emit-diagnostics`)

This document describes the JSON output format for the `apxm compile --emit-diagnostics` flag.

## Usage

```bash
apxm compile workflow.ais -O1 --emit-diagnostics diagnostics.json -o output.apxmobj
```

## Current Output (v1.0)

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

## Field Descriptions

### `input`
Absolute path to the input AIS file.

### `optimization_level`
Optimization level used: `O0`, `O1`, or `O2`.

### `compilation_phases`

| Field | Description |
|-------|-------------|
| `total_ms` | Total compilation time (parse + optimize + lower) |
| `artifact_gen_ms` | Time to generate the binary artifact |

### `dag_statistics`

| Field | Description |
|-------|-------------|
| `total_nodes` | Number of nodes in the execution DAG |
| `entry_nodes` | Nodes with no dependencies (can start immediately) |
| `exit_nodes` | Nodes with no consumers (produce final outputs) |
| `total_edges` | Number of data dependency edges |

### `passes_applied`

List of MLIR passes applied during optimization, in order.

## Enhanced Output (v2.0) - Target

Per-pass statistics will be written to the diagnostics JSON:

```json
{
  "input": "/path/to/workflow.ais",
  "optimization_level": "O1",
  "compilation_phases": {
    "total_ms": 9.87,
    "artifact_gen_ms": 0.19
  },
  "passes": [
    {
      "name": "normalize",
      "timing_ms": 0.82,
      "output": "Normalized 5 attributes (ctx_dedup=3, str_norm=2)",
      "statistics": {
        "graph_normalized": 5,
        "context_dedups": 3,
        "string_normalizations": 2
      }
    },
    {
      "name": "scheduling",
      "timing_ms": 1.24,
      "output": "Annotated 8 ops (inv=2, rsn=5, plan=1)",
      "statistics": {
        "scheduling_annotations": 8,
        "invocations": 2,
        "reasonings": 5,
        "plans": 1
      }
    },
    {
      "name": "fuse-reasoning",
      "timing_ms": 0.95,
      "output": "Scanned 5 RSN ops, fused 2 pairs",
      "statistics": {
        "scanned": 5,
        "fused_pairs": 2
      },
      "fusions": [
        {"producer": "rsn_0", "consumer": "rsn_1", "result": "rsn_fused_0"},
        {"producer": "rsn_2", "consumer": "rsn_3", "result": "rsn_fused_1"}
      ],
      "token_savings": {
        "estimated_tokens_saved": 500,
        "llm_calls_saved": 2
      }
    },
    {
      "name": "canonicalizer",
      "timing_ms": 0.41,
      "output": "Applied canonicalization patterns"
    },
    {
      "name": "cse",
      "timing_ms": 0.33,
      "output": "Eliminated 0 common subexpressions",
      "statistics": {"eliminated": 0}
    },
    {
      "name": "symbol-dce",
      "timing_ms": 0.28,
      "output": "Removed 0 dead symbols",
      "statistics": {"symbols_removed": 0}
    },
    {
      "name": "lower-to-async",
      "timing_ms": 0.62,
      "output": "Lowered to async execution model"
    }
  ],
  "token_estimation": {
    "method": "tiktoken (cl100k_base)",
    "original_tokens": 1250,
    "optimized_tokens": 750,
    "savings_pct": 40.0
  },
  "dag_statistics": {
    "total_nodes": 3,
    "entry_nodes": 1,
    "exit_nodes": 1,
    "total_edges": 2
  }
}
```

## Per-Pass Statistics

Statistics are collected from MLIR module attributes set by each pass:

| Pass | Attribute | Description |
|------|-----------|-------------|
| `normalize` | `ais.graph_normalized` | Number of normalized attributes |
| `scheduling` | `ais.scheduling_annotations` | Number of ops annotated |
| `fuse-reasoning` | `ais.fused_pairs` | Number of RSN pairs fused |

## Token Estimation

Token estimation uses tiktoken (cl100k_base encoding) to estimate:

- **Original tokens**: Total tokens if each RSN is a separate LLM call
- **Optimized tokens**: Total tokens after fusion (single call)
- **Savings**: Percentage reduction from fusion

## Implementation Status

| Feature | Status |
|---------|--------|
| Basic compilation phases | ✅ Implemented |
| DAG statistics | ✅ Implemented |
| Passes list | ✅ Implemented |
| Per-pass timing | 🔄 Planned |
| Per-pass statistics | 🔄 Planned (requires C++/FFI) |
| Token estimation | 🔄 Planned (requires tiktoken-rs) |

## Implementation Notes

Per-pass statistics require exposing MLIR module attributes via FFI:

```cpp
// In mlir/lib/CAPI/Attributes.cpp
int64_t aisModuleGetIntAttr(MlirModule module, const char* attrName) {
    auto attr = unwrap(module)->getAttrOfType<IntegerAttr>(attrName);
    return attr ? attr.getInt() : -1;
}
```

Then in Rust:

```rust
// In src/ffi/mod.rs
extern "C" {
    fn aisModuleGetIntAttr(module: MlirModule, name: *const c_char) -> i64;
}
```
