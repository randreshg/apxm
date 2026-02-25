# apxm-runtime

Production-ready execution engine for APXM programs.

## Overview

`apxm-runtime` executes compiled AIS programs using a dataflow scheduling model. It provides:
- **Three-tier memory system** (STM, LTM, Episodic)
- **Capability system** for tool/API integration
- **Dataflow scheduler** with work-stealing
- **Observability** via metrics and tracing

## Responsibilities

- Execute artifacts via the dataflow scheduler
- Manage AAM memory tiers (STM/LTM/Episodic)
- Dispatch AIS operations to handlers and capabilities

## How It Fits

`apxm-runtime` consumes artifacts produced by `apxm-compiler` (typically via
`apxm-driver`) and executes them using the dataflow scheduler and AAM state.
`apxm-backends` is treated as a runtime subsystem for LLM and storage access.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Runtime                               │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────┐  │
│  │                Runtime Agents (Agent/Flow)            │  │
│  └──────────────────────────┬────────────────────────────┘  │
│                             │                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Scheduler  │  │   Executor   │  │  Capability Sys  │  │
│  │  (Dataflow)  │  │   (DAG)      │  │   (Tools/APIs)   │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                 │                    │            │
│  ┌──────┴─────────────────┴────────────────────┴─────────┐  │
│  │                    AAM (Agent Abstract Machine)        │  │
│  │  ┌─────────┐  ┌─────────┐  ┌───────────┐              │  │
│  │  │   STM   │  │   LTM   │  │ Episodic  │              │  │
│  │  │ (Short) │  │ (Long)  │  │ (Traces)  │              │  │
│  │  └─────────┘  └─────────┘  └───────────┘              │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Key Types

- `Runtime` - Main runtime instance
- `RuntimeConfig` - Configuration for memory, scheduling
- `Agent` / `AgentFlow` - Runtime agent + flow hierarchy model
- `ExecutionDag` - Directed acyclic graph of operations
- `CapabilitySystem` - Registry for tools and APIs
- `MemorySystem` - STM/LTM/Episodic memory management

## Usage

```rust
use apxm_runtime::{Runtime, RuntimeConfig};
use apxm_artifact::Artifact;

// Create runtime with default config
let runtime = Runtime::new(RuntimeConfig::default()).await?;

// Load and execute artifact
let artifact = Artifact::from_bytes(&artifact_bytes)?;
let dag = artifact.into_dag();

let result = runtime.execute(dag).await?;
println!("Execution complete: {:?}", result.stats);
```

## Memory System

Three-tier Agent Abstract Machine (AAM) memory:

| Tier | Purpose | Persistence |
|------|---------|-------------|
| **STM** | Working memory for current task | Session |
| **LTM** | Long-term knowledge storage | Persistent |
| **Episodic** | Execution traces for reflection | Configurable |

```rust
use apxm_runtime::{MemoryConfig, RuntimeConfig};

let config = RuntimeConfig {
    memory_config: MemoryConfig::default(),
    ..Default::default()
};
```

## Capability System

Register tools and APIs for `inv` operations:

```rust
use apxm_runtime::capability::{CapabilitySystem, CapabilityRecord};

// Register a capability
runtime.capability_system().register(
    "web_search",
    CapabilityRecord::new("Search the web", my_search_fn),
)?;

// List available capabilities
let names = runtime.capability_system().list_capability_names();
```

For multi-flow artifacts, runtime agent registration uses
`FlowRegistry::register_agent(agent)` which stores the full runtime `Agent`
object and also backfills legacy `(agent, flow)` DAG lookups for `flow_call`.

## Scheduler

Work-stealing dataflow scheduler with configurable parallelism:

```rust
use apxm_runtime::{RuntimeConfig, SchedulerConfig};

let config = RuntimeConfig {
    scheduler_config: SchedulerConfig {
        max_parallelism: 8,
        enable_work_stealing: true,
    },
    ..Default::default()
};
```

## Metrics (Optional)

Enable LLM usage and runtime timing metrics with:

```bash
cargo test -p apxm-runtime --features metrics
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| apxm-core | Core types (DAG, Node, Value) |
| apxm-backends | LLM providers, storage backends |
| apxm-artifact | Artifact deserialization |

## Building

```bash
cargo build -p apxm-runtime
```

## Testing

```bash
cargo test -p apxm-runtime
```

## Module Structure

```
crates/runtime/apxm-runtime/src/
├── lib.rs
├── runtime.rs          # Main Runtime struct
├── aam/                # Agent Abstract Machine
├── capability/         # Tool/API registration
├── executor/           # DAG execution engine
│   ├── handlers/       # Operation handlers (20+)
│   └── dispatcher.rs   # Operation dispatch
├── memory/             # STM/LTM/Episodic
├── scheduler/          # Dataflow scheduler
│   ├── dataflow.rs
│   ├── work_stealing.rs
│   └── splicing.rs
└── observability/      # Metrics and tracing
```
