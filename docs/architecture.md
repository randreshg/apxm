# APXM Architecture Documentation

This document provides a comprehensive overview of how the different packages in APXM interact with each other, including flowcharts, diagrams, and class relationships.

## Table of Contents

1. [System Overview](#system-overview)
2. [Package Structure](#package-structure)
3. [Driver Flows](#driver-flows)
4. [Compilation Pipeline](#compilation-pipeline)
5. [Runtime Execution](#runtime-execution)
7. [Class Relationships](#class-relationships)

---

## System Overview

APXM (Agent Programming eXecution Model) is a full toolchain for building autonomous agents that combines:
- A high-level DSL (AIS) for declaring memory, flows, handlers, and tool invocations
- A compiler that lowers AIS → MLIR → executable artifacts
- A runtime/linker that wires artifacts to capabilities, an LLM registry, and execution memory
- A driver crate that wires compiler + runtime
- A minimal CLI for compile/run workflows

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Driver                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│  │ compile  │  │   run    │  │   link   │                     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                     │
└───────┼─────────────┼─────────────┼──────────────────────────────┘
        │             │             │
        ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Linker Layer                               │
│  ┌──────────────┐         ┌──────────────┐                    │
│  │   Compiler   │────────▶│   Runtime    │                    │
│  │  (MLIR/C++)  │         │  Executor    │                    │
│  └──────────────┘         └──────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
        │                                     │
        ▼                                     ▼
┌──────────────────┐              ┌──────────────────────────────┐
│  Artifact        │              │  Runtime Components         │
│  (Binary Format) │              │  - Memory System            │
│                  │              │  - Scheduler                │
│  - DAG           │              │  - Executor Engine          │
│  - Metadata      │              │  - Capability System        │
│  - Sections      │              │  - LLM Registry             │
└──────────────────┘              └──────────────────────────────┘
```

---

## Package Structure

### Core Packages

- **`apxm-ais`**: AIS operation definitions and validation
- **`apxm-compiler`**: MLIR-based compiler (C++/Rust FFI)
- **`apxm-runtime`**: Execution engine with scheduler, memory, capabilities
- **`apxm-driver`**: Orchestrates compiler + runtime and config
- **`apxm-backends`**: LLM and storage backends
- **`apxm-core`**: Shared types, errors, and utilities
- **`apxm-artifact`**: Binary artifact format (serialized DAGs)
- **`apxm-ais`**: AIS metadata and TableGen source

### Package Dependencies

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

---

## Driver Flows

### 1. Compile Command Flow

```mermaid
graph TD
    A[apxm-driver compile()] --> B[Create Compiler Context]
    B --> C{Input Type?}
    C -->|.ais| D[Parse DSL]
    C -->|.mlir| E[Parse MLIR]
    D --> F[Module Object]
    E --> F
    F --> G[Build Pass Manager]
    G --> H[Run Optimization Passes]
    H --> I{Output Format?}
    I -->|Artifact| J[Generate Artifact]
    I -->|Rust| K[Generate Rust Code]
    I -->|MLIR| L[Dump MLIR Text]
    J --> M[Return Bytes]
    K --> M
    L --> M
```

**Key Components:**
- **`Context`**: MLIR compiler context (C++ FFI)
- **`Module`**: MLIR module representation
- **`PassManager`**: Orchestrates optimization passes
- **`Codegen`**: Emits artifact or Rust source

### 2. Run Command Flow

```mermaid
graph TD
    A[apxm-driver execute()] --> B[Load Config]
    B --> C[Create RuntimeExecutor]
    C --> D[Compiler.compile]
    D --> E[Module.generate_artifact_bytes]
    E --> F[Artifact.from_bytes]
    F --> G[Validate DAG]
    G --> H[RuntimeExecutor.execute]
    H --> I[Runtime.execute]
    I --> J[DataflowScheduler.execute]
    J --> K[ExecutorEngine.execute_dag]
    K --> L[OperationDispatcher.dispatch]
    L --> M[Return Results]
```

**Key Components:**
- **`Compiler`**: Wraps MLIR compiler
- **`RuntimeExecutor`**: Wraps runtime with LLM/capability setup
- **`Runtime`**: Main runtime orchestrator
- **`DataflowScheduler`**: Parallel execution scheduler
- **`ExecutorEngine`**: Operation execution engine

---

## Compilation Pipeline

### Compilation Stages

```mermaid
graph LR
    A[DSL Source] --> B[Parser]
    B --> C[MLIR Module]
    C --> D[Pass Manager]
    D --> E[Optimization Passes]
    E --> F[Lowering Passes]
    F --> G[Final MLIR]
    G --> H[Code Generation]
    H --> I[Artifact]
    
    style A fill:#e1f5ff
    style C fill:#fff4e1
    style G fill:#fff4e1
    style I fill:#e8f5e9
```

### Optimization Pipeline (by -O level)

**O0 (No optimization):**
- `canonicalizer` → `cse`

**O1 (Basic):**
- `normalize` → `scheduling` → `fuse-reasoning` → `canonicalizer` → `cse` → `symbol-dce`

**O2 (Standard):**
- O1 pipeline + second `fuse-reasoning` pass + `canonicalizer`

**O3 (Aggressive):**
- Multiple rounds of fusion, scheduling, and cleanup passes

### Artifact Generation

```mermaid
graph TD
    A[MLIR Module] --> B[Generate Artifact Bytes]
    B --> C[Serialize DAG]
    C --> D[Add Metadata]
    D --> E[Add Sections]
    E --> F[Calculate Hash]
    F --> G[Write Binary Format]
    G --> H[.apxmobj File]
    
    H --> I[Header: Magic + Version]
    I --> J[Payload: DAG + Metadata]
    J --> K[Hash: BLAKE3]
```

**Artifact Structure:**
- **Header**: Magic bytes (`APXM`), version, timestamp, hash, payload size
- **Payload**: Serialized DAG (nodes, edges, metadata)
- **Sections**: Optional additional data sections

---

## Runtime Execution

### Runtime Architecture

```mermaid
graph TB
    subgraph "Runtime Components"
        A[Runtime]
        B[MemorySystem]
        C[LLMRegistry]
        D[CapabilitySystem]
        E[AAM]
        F[DataflowScheduler]
        G[ExecutorEngine]
    end
    
    A --> B
    A --> C
    A --> D
    A --> E
    A --> F
    F --> G
    
    G --> H[OperationDispatcher]
    H --> I[Operation Handlers]
    I --> J[const_str<br/>umem/qmem<br/>invoke<br/>communicate<br/>plan<br/>reflect<br/>...]
```

### Execution Flow

```mermaid
sequenceDiagram
    participant Linker
    participant Runtime
    participant Scheduler
    participant Executor
    participant Dispatcher
    participant Handler
    
    Linker->>Runtime: execute(dag)
    Runtime->>Runtime: Create ExecutionContext
    Runtime->>Scheduler: execute(dag, executor, context)
    Scheduler->>Scheduler: Build Ready Set
    Scheduler->>Executor: execute_node(node, inputs)
    Executor->>Dispatcher: dispatch(context, node, inputs)
    Dispatcher->>Handler: handle(node.op_type)
    Handler-->>Dispatcher: Value
    Dispatcher-->>Executor: Value
    Executor-->>Scheduler: Result
    Scheduler->>Scheduler: Update Dependencies
    Scheduler-->>Runtime: (results, stats)
    Runtime-->>Linker: RuntimeExecutionResult
```

### Memory System

```
┌─────────────────────────────────────────┐
│         MemorySystem                    │
├─────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌────────┐│
│  │   STM    │  │   LTM    │  │Episodic││
│  │(Working) │  │(Persist) │  │(Events)││
│  └──────────┘  └──────────┘  └────────┘│
└─────────────────────────────────────────┘
```

- **STM (Short-Term Memory)**: Working memory for current execution
- **LTM (Long-Term Memory)**: Persistent storage (SQLite or in-memory)
- **Episodic**: Event/experience storage

### Scheduler Architecture

```mermaid
graph TD
    A[DataflowScheduler] --> B[ReadySet]
    A --> C[PriorityQueue]
    A --> D[Worker Pool]
    A --> E[ConcurrencyControl]
    
    B --> F[Track Ready Nodes]
    C --> G[4 Priority Levels]
    D --> H[Work Stealing]
    E --> I[Semaphore Backpressure]
    
    G --> J[Critical<br/>High<br/>Normal<br/>Low]
```

**Features:**
- Token-based automatic parallelism
- O(1) readiness tracking
- Exponential backoff retry
- Deadlock detection
- Semaphore-based backpressure

---

## Class Relationships

### Linker Package

```mermaid
classDiagram
    class Linker {
        -Compiler compiler
        -RuntimeExecutor runtime
        +new(config) Linker
        +compile_only(input, mlir) Module
        +run(input, mlir) LinkResult
        +runtime_llm_registry() LLMRegistry
        +runtime_capabilities() Vec~String~
    }
    
    class Compiler {
        +compile(input, mlir) Module
    }
    
    class RuntimeExecutor {
        -Runtime runtime
        +new(config) RuntimeExecutor
        +execute(dag) RuntimeExecutionResult
        +llm_registry() LLMRegistry
        +capability_names() Vec~String~
    }
    
    class LinkResult {
        +Module module
        +Artifact artifact
        +RuntimeExecutionResult execution
    }
    
    Linker --> Compiler
    Linker --> RuntimeExecutor
    Linker --> LinkResult
    RuntimeExecutor --> Runtime
```

### Runtime Package

```mermaid
classDiagram
    class Runtime {
        -MemorySystem memory
        -LLMRegistry llm_registry
        -CapabilitySystem capability_system
        -Aam aam
        -DataflowScheduler scheduler
        -InnerPlanLinker inner_plan_linker
        +new(config) Runtime
        +execute(dag) RuntimeExecutionResult
        +execute_artifact(artifact) RuntimeExecutionResult
    }
    
    class MemorySystem {
        +read(space, key) Value
        +write(space, key, value)
        +stm() STM
        +ltm() LTM
    }
    
    class DataflowScheduler {
        +execute(dag, executor, context) (results, stats)
    }
    
    class ExecutorEngine {
        -ExecutionContext context
        +execute_dag(dag) ExecutionResult
        +execute_node(node, inputs) Value
    }
    
    class ExecutionContext {
        +MemorySystem memory
        +LLMRegistry llm_registry
        +CapabilitySystem capabilities
        +Aam aam
        +InnerPlanLinker inner_plan_linker
    }
    
    class OperationDispatcher {
        +dispatch(context, node, inputs) Value
    }
    
    Runtime --> MemorySystem
    Runtime --> DataflowScheduler
    Runtime --> ExecutorEngine
    ExecutorEngine --> ExecutionContext
    ExecutorEngine --> OperationDispatcher
```

### Core Types

```mermaid
classDiagram
    class ExecutionDag {
        +Vec~Node~ nodes
        +Vec~Edge~ edges
        +Vec~u64~ entry_nodes
        +Vec~u64~ exit_nodes
        +validate() Result
    }
    
    class Node {
        +u64 id
        +AISOperationType op_type
        +HashMap attributes
        +Vec~u64~ input_tokens
        +Vec~u64~ output_tokens
    }
    
    class Artifact {
        -ArtifactMetadata metadata
        -ExecutionDag dag
        -Vec~ArtifactSection~ sections
        +dag() ExecutionDag
        +into_dag() ExecutionDag
        +to_bytes() Vec~u8~
        +from_bytes(bytes) Artifact
    }
    
    class Value {
        <<enumeration>>
        String
        Number
        Boolean
        Null
        Array
        Object
    }
    
    ExecutionDag --> Node
    Artifact --> ExecutionDag
    Node --> Value
```

---

## Data Flow Summary

### Complete Execution Flow (Run Command)

```
User Input (DSL/MLIR)
    ↓
Linker.run()
    ↓
Compiler.compile() → Module
    ↓
Module.generate_artifact_bytes() → Vec<u8>
    ↓
Artifact.from_bytes() → Artifact
    ↓
Artifact.dag() → ExecutionDag
    ↓
Runtime.execute(dag)
    ↓
DataflowScheduler.execute(dag, executor, context)
    ↓
ExecutorEngine.execute_node() (for each ready node)
    ↓
OperationDispatcher.dispatch()
    ↓
Operation Handler (const_str, invoke, communicate, etc.)
    ↓
Value (result)
    ↓
RuntimeExecutionResult { results, stats }
```

## Key Design Patterns

1. **Orchestration Pattern**: `Linker` orchestrates `Compiler` and `Runtime`
2. **Strategy Pattern**: Different operation handlers via `OperationDispatcher`
3. **Factory Pattern**: `LLMRegistry` creates backend instances
4. **Builder Pattern**: Configuration builders for `RuntimeConfig`, `LinkerConfig`

---

## Error Handling

Errors flow through the system with proper error types:

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

---

## Configuration Flow

```
Project .apxm/config.toml (walking up directories)
    ↓
Global ~/.apxm/config.toml
    ↓
ApXmConfig (TOML deserialization)
    ↓
RuntimeConfig
    ↓
Component initialization
```

---

This architecture enables:
- **Separation of Concerns**: Each package has a clear responsibility
- **Composability**: Components can be used independently
- **Extensibility**: New operation types, LLM backends, and capabilities can be added
- **Testability**: Each layer can be tested in isolation
- **Performance**: Parallel execution via dataflow scheduler
- **Reliability**: Comprehensive error handling and validation
