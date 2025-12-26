# APXM Architecture Summary

Quick reference guide for understanding APXM's architecture and component interactions.

## Documentation Files

- `docs/architecture.md`: Full architecture walkthrough
- `docs/diagrams.md`: Visual flowcharts and diagrams
- `docs/class_relationships.md`: Class relationship diagrams
- `docs/architecture_summary.md`: This file

## Compile Flow (apxm-compiler)

```
DSL/MLIR → Parser → MLIR Module → Pass Manager → Optimized MLIR → Artifact
```

Key locations:
- `crates/apxm-compiler/`
- `crates/apxm-driver/src/compiler/`

## Execute Flow (apxm-driver + apxm-runtime)

```
Input → Driver.compile() → Artifact.from_bytes() → ExecutionDag → Runtime.execute()
```

Key locations:
- `crates/apxm-driver/src/linker/mod.rs`
- `crates/runtime/apxm-runtime/src/runtime.rs`
- `crates/runtime/apxm-runtime/src/executor/engine.rs`

## Package Responsibilities

| Package | Responsibility | Key Types |
|---------|---------------|-----------|
| `apxm-ais` | AIS operation definitions + validation | `AISOperationType`, `OperationSpec` |
| `apxm-compiler` | MLIR compilation (C++ FFI) | `Context`, `Module`, `PassManager` |
| `apxm-driver` | Orchestrates compiler + runtime | `RuntimeExecutor`, `Linker` |
| `apxm-runtime` | Execution engine | `Runtime`, `DataflowScheduler`, `ExecutorEngine` |
| `apxm-backends` | Runtime subsystem: LLM + storage backends | `LLMRegistry`, `LLMBackend` |
| `apxm-core` | Shared types and errors | `ExecutionDag`, `Node`, `Value`, `RuntimeError` |
| `apxm-artifact` | Binary artifact format | `Artifact`, `ArtifactMetadata` |
| `apxm-ais` | AIS metadata + TableGen | `OperationMetadata` |
| `apxm-cli` | Minimal compile/run tooling | `apxm` binary |

## Memory System

```
MemorySystem
├── STM (Short-Term Memory)
├── LTM (Long-Term Memory)
└── Episodic (Execution traces)
```

## Notes

- `COMMUNICATE` remains stubbed in the runtime; document this in the paper.
- Metrics/instrumentation can be compiled out for zero runtime overhead.
