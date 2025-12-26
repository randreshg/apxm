# APXM Quick Reference

Visual quick reference for understanding APXM's architecture at a glance.

## Main Execution Paths

### Path 1: Compile Only
```
┌─────────────┐
│ User Input  │ (DSL/MLIR file)
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  CLI/Driver     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   Compiler      │ Parse → Optimize → Lower
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   Output        │ Artifact / Rust / MLIR
└─────────────────┘
```

### Path 2: Compile + Run
```
┌─────────────┐
│ User Input  │ (DSL/MLIR file)
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  CLI/Driver     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐      ┌─────────────────┐
│   Driver         │─────▶│   Compiler       │
└──────┬───────────┘      └────────┬─────────┘
       │                            │
       │                            ▼
       │                   ┌─────────────────┐
       │                   │   Artifact      │
       │                   └────────┬────────┘
       │                            │
       │                            ▼
       │                   ┌─────────────────┐
       │                   │  ExecutionDag    │
       │                   └────────┬────────┘
       │                            │
       └────────────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │   Runtime            │
         │  ┌────────────────┐ │
         │  │   Scheduler    │ │
         │  └────────┬───────┘ │
         │           │         │
         │           ▼         │
         │  ┌────────────────┐ │
         │  │   Executor     │ │
         │  └────────┬───────┘ │
         │           │         │
         │           ▼         │
         │  ┌────────────────┐ │
         │  │  Dispatcher   │ │
         │  └────────┬───────┘ │
         │           │         │
         │           ▼         │
         │  ┌────────────────┐ │
         │  │   Handlers     │ │
         │  └────────────────┘ │
         └──────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │   Results            │
         └──────────────────────┘
```

## Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         Driver                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │ compile  │  │   run    │  │   link   │                 │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                 │
└───────┼─────────────┼─────────────┼─────────────────────────┘
        │             │             │
        ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────┐
│                      Compiler/Runtime                       │
│  ┌──────────────┐         ┌──────────────┐                │
│  │  Compiler    │────────▶│   Runtime    │                │
│  │  (MLIR/C++)  │         │  Executor    │                │
│  └──────────────┘         └──────────────┘                │
└─────────────────────────────────────────────────────────────┘
        │                                     │
        ▼                                     ▼
┌──────────────────┐              ┌──────────────────────────┐
│   Artifact       │              │   Runtime Components     │
│                  │              │                          │
│  ┌────────────┐ │              │  ┌────────────────────┐ │
│  │    DAG     │ │              │  │  MemorySystem      │ │
│  │  (Nodes,   │ │              │  │  - STM             │ │
│  │   Edges)   │ │              │  │  - LTM             │ │
│  └────────────┘ │              │  │  - Episodic       │ │
│                 │              │  └────────────────────┘ │
│  ┌────────────┐ │              │                        │
│  │ Metadata   │ │              │  ┌────────────────────┐ │
│  └────────────┘ │              │  │  LLMRegistry      │ │
│                 │              │  │  - OpenAI          │ │
│  ┌────────────┐ │              │  │  - Anthropic      │ │
│  │ Sections   │ │              │  │  - Google         │ │
│  └────────────┘ │              │  │  - Ollama        │ │
└──────────────────┘              │  └────────────────────┘ │
                                   │                        │
                                   │  ┌────────────────────┐ │
                                   │  │ CapabilitySystem   │ │
                                   │  └────────────────────┘ │
                                   │                        │
                                   │  ┌────────────────────┐ │
                                   │  │ DataflowScheduler  │ │
                                   │  │  - ReadySet        │ │
                                   │  │  - PriorityQueue   │ │
                                   │  │  - Workers         │ │
                                   │  └────────────────────┘ │
                                   │                        │
                                   │  ┌────────────────────┐ │
                                   │  │ ExecutorEngine    │ │
                                   │  │  - Dispatcher      │ │
                                   │  │  - Handlers       │ │
                                   │  └────────────────────┘ │
                                   └──────────────────────────┘
```

## Data Flow Summary

### Compilation Flow
```
DSL/MLIR → Parser → MLIR Module → Pass Manager → Optimized MLIR → Artifact
```

### Execution Flow
```
Artifact → ExecutionDag → Scheduler → Executor → Dispatcher → Handler → Value
```

### Translation Flow (optional)
```
LLM Plan → DSL → Validate → Compile → Execute
```

## Package Dependencies

```
apxm-driver
├── apxm-compiler
├── runtime
│   ├── apxm-runtime
│   ├── apxm-core
│   ├── apxm-backends
│   └── apxm-artifact
└── apxm-ais
```

## Memory Hierarchy

```
MemorySystem
│
├── STM (Short-Term Memory)
│   └── Working memory (HashMap)
│
├── LTM (Long-Term Memory)
│   ├── SQLite (persistent)
│   └── In-Memory (testing)
│
└── Episodic Memory
    └── Append-only in-memory log
```

## Scheduler Architecture

```
DataflowScheduler
│
├── ReadySet
│   └── Tracks nodes ready to execute
│
├── PriorityQueue
│   ├── Critical
│   ├── High
│   ├── Normal
│   └── Low
│
├── Worker Pool
│   └── Work-stealing threads
│
└── ConcurrencyControl
    └── Semaphore-based backpressure
```

## Operation Types Quick Reference

| Type | Purpose | Handler |
|------|---------|---------|
| `CONST_STR` | Constant values | ConstStrHandler |
| `UMEM` | Write memory | UMemHandler |
| `QMEM` | Query memory | QMemHandler |
| `INVOKE` | Call capability | InvokeHandler |
| `COMMUNICATE` | LLM call | CommunicateHandler |
| `PLAN` | Generate plan | PlanHandler |
| `REFLECT` | Reasoning | ReflectHandler |
| `BRANCH` | Conditionals | BranchHandler |
| `LOOP_*` | Loops | LoopHandler |
| `MERGE` | Merge branches | MergeHandler |
| `WAIT_ALL` | Sync | WaitAllHandler |
| `TRY_CATCH` | Error handling | TryCatchHandler |

## Configuration Priority

```
1. Project .apxm/config.toml (walking up)
   │
2. Global ~/.apxm/config.toml
   │
3. Default config (lowest)
```

## Error Types

```
DriverError
├── Compiler(CompilerError)
├── Runtime(RuntimeError)
├── Config { message }
└── Io(io::Error)

RuntimeError
├── State(String)
├── Execution(String)
└── Memory(String)
```

## Common Debugging Commands

```bash
# Run runtime examples
cargo run -p apxm-runtime --example substrate_demo
cargo run -p apxm-runtime --example ollama_llm_demo

```

## File Locations

| Component | Location |
|-----------|----------|
| Driver | `crates/apxm-driver/src/` |
| Runtime | `crates/runtime/apxm-runtime/src/runtime.rs` |
| Scheduler | `crates/runtime/apxm-runtime/src/scheduler/` |
| Executor | `crates/runtime/apxm-runtime/src/executor/` |
| Core Types | `crates/runtime/apxm-core/src/types/` |
| Artifact | `crates/runtime/apxm-artifact/src/` |

## Quick Troubleshooting

| Issue | Check |
|-------|-------|
| Compilation fails | Check DSL syntax, see compiler errors |
| Execution hangs | Check for deadlocks, use verbose logging |
| Memory issues | Check LTM backend (SQLite vs in-memory) |
| Slow execution | Check scheduler configuration, concurrency limits |

---

For detailed information, see:
- **docs/architecture.md**: Comprehensive architecture documentation
- **docs/diagrams.md**: Visual flowcharts and diagrams
- **docs/class_relationships.md**: Class relationship diagrams
- **docs/architecture_summary.md**: Detailed summary with examples
