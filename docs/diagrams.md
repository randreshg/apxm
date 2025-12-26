# APXM Visual Diagrams

This file contains detailed flowcharts and diagrams for understanding APXM's architecture.

## 0. Workspace Layout

```mermaid
graph TD
    ROOT[workspace root]
    CRATES[crates/]
    RUNTIME[crates/runtime/]

    ROOT --> CRATES
    ROOT --> RUNTIME

    CRATES --> AIS[apxm-ais]
    CRATES --> CLI[apxm-cli]
    CRATES --> COMPILER[apxm-compiler]
    CRATES --> DRIVER[apxm-driver]
    CRATES --> SHARED[apxm-ais]

    RUNTIME --> CORE[apxm-core]
    RUNTIME --> BACKENDS[apxm-backends]
    RUNTIME --> ARTIFACT[apxm-artifact]
    RUNTIME --> RT[apxm-runtime]
```

## 1. Complete System Flow

```mermaid
graph TB
    subgraph "User Interface"
        CLI[apxm-cli]
        DRIVER_API[Driver API / Examples]
    end
    
    subgraph "Driver Layer"
        COMPILE[compile]
        RUN[run]
        COMPILER_WRAPPER[Compiler Wrapper]
        DRIVER[RuntimeExecutor]
    end
    
    subgraph "Compiler Layer"
        MLIR_CTX[MLIR Context]
        MODULE[MLIR Module]
        PASSES[Pass Manager]
    end
    
    subgraph "Runtime Layer"
        RUNTIME[Runtime]
        SCHEDULER[DataflowScheduler]
        EXECUTOR[ExecutorEngine]
        DISPATCHER[OperationDispatcher]
    end
    
    subgraph "Supporting Systems"
        MEMORY[MemorySystem]
        LLM_REG[LLMRegistry]
        CAPS[CapabilitySystem]
        AAM[AAM]
    end
    
    CLI --> COMPILE
    CLI --> RUN
    DRIVER_API --> COMPILE
    DRIVER_API --> RUN
    
    COMPILE --> COMPILER_WRAPPER
    RUN --> DRIVER
    
    COMPILER_WRAPPER --> MLIR_CTX
    MLIR_CTX --> MODULE
    MODULE --> PASSES
    PASSES --> ARTIFACT[Artifact]
    
    DRIVER --> RUNTIME
    RUNTIME --> SCHEDULER
    SCHEDULER --> EXECUTOR
    EXECUTOR --> DISPATCHER
    
    RUNTIME --> MEMORY
    RUNTIME --> LLM_REG
    RUNTIME --> CAPS
    RUNTIME --> AAM
    
    ARTIFACT --> RUNTIME
    
    style DRIVER_API fill:#e1f5ff
    style DRIVER fill:#fff4e1
    style RUNTIME fill:#e8f5e9
    style ARTIFACT fill:#f3e5f5
```

## 2. Compilation Pipeline Detail

```mermaid
graph LR
    subgraph "Input"
        DSL[DSL Source<br/>.ais file]
        MLIR_IN[MLIR Source<br/>.mlir file]
    end
    
    subgraph "Parsing"
        PARSE_DSL[Parse DSL]
        PARSE_MLIR[Parse MLIR]
    end
    
    subgraph "MLIR Module"
        MODULE[MLIR Module<br/>Internal Representation]
    end
    
    subgraph "Optimization"
        NORM[Normalize]
        SCHED[Scheduling]
        FUSE[Fuse Reasoning]
        CANON[Canonicalizer]
        CSE[Common Subexpression<br/>Elimination]
        DCE[Symbol DCE]
    end
    
    subgraph "Lowering"
        INLINE[Inline]
        LOWER[Lower to Async]
    end
    
    subgraph "Code Generation"
        GEN_ARTIFACT[Generate Artifact]
        GEN_RUST[Generate Rust]
        DUMP_MLIR[Dump MLIR]
    end
    
    DSL --> PARSE_DSL
    MLIR_IN --> PARSE_MLIR
    PARSE_DSL --> MODULE
    PARSE_MLIR --> MODULE
    
    MODULE --> NORM
    NORM --> SCHED
    SCHED --> FUSE
    FUSE --> CANON
    CANON --> CSE
    CSE --> DCE
    
    DCE --> INLINE
    INLINE --> LOWER
    
    LOWER --> GEN_ARTIFACT
    LOWER --> GEN_RUST
    LOWER --> DUMP_MLIR
    
    style MODULE fill:#fff4e1
    style GEN_ARTIFACT fill:#e8f5e9
```

## 2.5 Shared Ops + End-to-End Flow

```mermaid
graph LR
    subgraph "Shared Operation Definitions"
        SHARED[apxm-ais (Rust ops)]
        TBLGEN[TableGen generator]
        TD[AIS .td files]
        RUST_META[Rust metadata]
    end

    subgraph "Compiler"
        DSL[AIS DSL]
        FRONT[Lexer/Parser/AST]
        MLIRGEN[MLIRGen]
        AISMLIR[AIS MLIR]
        PASSES[Pass Pipeline]
        ARTIFACT[Artifact]
    end

    subgraph "Runtime"
        RUNTIME[Runtime]
        SCHED[Dataflow Scheduler]
    end

    SHARED --> TBLGEN --> TD
    SHARED --> RUST_META

    DSL --> FRONT --> MLIRGEN --> AISMLIR --> PASSES --> ARTIFACT
    TD --> AISMLIR

    ARTIFACT --> RUNTIME
    RUST_META --> RUNTIME
    RUNTIME --> SCHED
```

## 3. Runtime Execution Detail

```mermaid
graph TD
    subgraph "Entry Point"
        DRIVER_RUN[Driver.execute]
    end
    
    subgraph "Compilation Phase"
        COMPILE[Compiler.compile]
        GEN_BYTES[Module.generate_artifact_bytes]
        PARSE_ARTIFACT[Artifact.from_bytes]
        VALIDATE[Validate DAG]
    end
    
    subgraph "Runtime Initialization"
        CREATE_RUNTIME[Runtime.new]
        INIT_MEMORY[Initialize MemorySystem]
        INIT_LLM[Initialize LLMRegistry]
        INIT_CAPS[Initialize CapabilitySystem]
        INIT_SCHED[Initialize Scheduler]
    end
    
    subgraph "Execution Phase"
        RUNTIME_EXEC[Runtime.execute]
        CREATE_CTX[Create ExecutionContext]
        SCHED_EXEC[Scheduler.execute]
        BUILD_READY[Build Ready Set]
    end
    
    subgraph "Node Execution Loop"
        GET_READY[Get Ready Nodes]
        EXEC_NODE[ExecutorEngine.execute_node]
        DISPATCH[OperationDispatcher.dispatch]
        HANDLER[Operation Handler]
        UPDATE_DEPS[Update Dependencies]
    end
    
    subgraph "Operation Handlers"
        CONST[const_str]
        MEM[umem/qmem]
        INVOKE[inv]
        COMM[communicate]
        PLAN[plan]
        REFLECT[reflect]
        BRANCH[branch]
        LOOP[loop_start/end]
    end
    
    DRIVER_RUN --> COMPILE
    COMPILE --> GEN_BYTES
    GEN_BYTES --> PARSE_ARTIFACT
    PARSE_ARTIFACT --> VALIDATE
    VALIDATE --> CREATE_RUNTIME
    
    CREATE_RUNTIME --> INIT_MEMORY
    CREATE_RUNTIME --> INIT_LLM
    CREATE_RUNTIME --> INIT_CAPS
    CREATE_RUNTIME --> INIT_SCHED
    
    INIT_SCHED --> RUNTIME_EXEC
    RUNTIME_EXEC --> CREATE_CTX
    CREATE_CTX --> SCHED_EXEC
    SCHED_EXEC --> BUILD_READY
    
    BUILD_READY --> GET_READY
    GET_READY --> EXEC_NODE
    EXEC_NODE --> DISPATCH
    DISPATCH --> HANDLER
    
    HANDLER --> CONST
    HANDLER --> MEM
    HANDLER --> INVOKE
    HANDLER --> COMM
    HANDLER --> PLAN
    HANDLER --> REFLECT
    HANDLER --> BRANCH
    HANDLER --> LOOP
    
    CONST --> UPDATE_DEPS
    MEM --> UPDATE_DEPS
    INVOKE --> UPDATE_DEPS
    COMM --> UPDATE_DEPS
    PLAN --> UPDATE_DEPS
    REFLECT --> UPDATE_DEPS
    BRANCH --> UPDATE_DEPS
    LOOP --> UPDATE_DEPS
    
    UPDATE_DEPS --> GET_READY
    
    style RUNTIME_EXEC fill:#e8f5e9
    style SCHED_EXEC fill:#fff4e1
    style HANDLER fill:#e1f5ff
```

## 4. Memory System Architecture

```mermaid
graph TB
    subgraph "MemorySystem"
        MS[MemorySystem]
        
        subgraph "Memory Spaces"
            STM[STM<br/>Short-Term Memory<br/>Working Memory]
            LTM[LTM<br/>Long-Term Memory<br/>Persistent Storage]
            EPI[Episodic<br/>Event Storage]
        end
        
        subgraph "Storage Backends"
            INMEM[In-Memory<br/>HashMap]
            SQLITE[SQLite<br/>Persistent DB]
        end
    end
    
    MS --> STM
    MS --> LTM
    MS --> EPI
    
    STM --> INMEM
    LTM --> SQLITE
    LTM --> INMEM
    EPI --> SQLITE
    
    style STM fill:#e1f5ff
    style LTM fill:#fff4e1
    style EPI fill:#f3e5f5
```

## 6. Scheduler Workflow

```mermaid
graph TD
    START[Start Execution] --> INIT[Initialize Scheduler]
    INIT --> BUILD[Build Dependency Graph]
    BUILD --> READY[Identify Ready Nodes]
    
    READY --> QUEUE{Ready Nodes<br/>Available?}
    
    QUEUE -->|Yes| PRIORITY[Add to Priority Queue]
    QUEUE -->|No| CHECK{All Nodes<br/>Complete?}
    
    PRIORITY --> WORKER[Worker Picks Node]
    WORKER --> EXEC[Execute Node]
    
    EXEC --> SUCCESS{Success?}
    SUCCESS -->|Yes| UPDATE[Update Dependencies]
    SUCCESS -->|No| RETRY{Retry<br/>Allowed?}
    
    RETRY -->|Yes| BACKOFF[Exponential Backoff]
    BACKOFF --> QUEUE
    RETRY -->|No| FAIL[Mark Failed]
    
    UPDATE --> READY
    FAIL --> CHECK
    
    CHECK -->|No| DEADLOCK{Deadlock<br/>Detected?}
    CHECK -->|Yes| DONE[Return Results]
    
    DEADLOCK -->|Yes| ERROR[Deadlock Error]
    DEADLOCK -->|No| WAIT[Wait for Dependencies]
    WAIT --> READY
    
    style START fill:#e8f5e9
    style DONE fill:#e8f5e9
    style ERROR fill:#ffebee
```

## 7. Operation Handler Dispatch

```mermaid
graph TD
    DISPATCH[OperationDispatcher.dispatch] --> CHECK{Operation Type}
    
    CHECK -->|CONST_STR| CONST[ConstStrHandler]
    CHECK -->|UMEM| UMEM[UMemHandler]
    CHECK -->|QMEM| QMEM[QMemHandler]
    CHECK -->|INVOKE| INVOKE[InvokeHandler]
    CHECK -->|COMMUNICATE| COMM[CommunicateHandler]
    CHECK -->|PLAN| PLAN[PlanHandler]
    CHECK -->|REFLECT| REFLECT[ReflectHandler]
    CHECK -->|BRANCH| BRANCH[BranchHandler]
    CHECK -->|LOOP_START| LOOP_S[LoopStartHandler]
    CHECK -->|LOOP_END| LOOP_E[LoopEndHandler]
    CHECK -->|MERGE| MERGE[MergeHandler]
    CHECK -->|WAIT_ALL| WAIT[WaitAllHandler]
    CHECK -->|TRY_CATCH| TRY[TryCatchHandler]
    CHECK -->|RETURN| RET[ReturnHandler]
    
    CONST --> MEMORY[Access Memory]
    UMEM --> MEMORY
    QMEM --> MEMORY
    INVOKE --> CAPS[Call Capability]
    COMM --> LLM[Call LLM]
    PLAN --> LLM
    REFLECT --> LLM
    BRANCH --> COND[Evaluate Condition]
    LOOP_S --> COND
    LOOP_E --> COND
    
    MEMORY --> VALUE[Return Value]
    CAPS --> VALUE
    LLM --> VALUE
    COND --> VALUE
    
    VALUE --> DISPATCH
    
    style DISPATCH fill:#e1f5ff
    style VALUE fill:#e8f5e9
```

## 8. Artifact Format Structure

```mermaid
graph TB
    subgraph "Artifact File"
        HEADER[Header<br/>16 bytes]
        PAYLOAD[Payload<br/>Variable Size]
    end
    
    subgraph "Header Fields"
        MAGIC[Magic: APXM<br/>4 bytes]
        VERSION[Version: u32<br/>4 bytes]
        TIMESTAMP[Timestamp: u64<br/>8 bytes]
        HASH[Hash: BLAKE3<br/>32 bytes]
        SIZE[Payload Size: u32<br/>4 bytes]
    end
    
    subgraph "Payload Structure"
        METADATA[ArtifactMetadata]
        DAG[ExecutionDag]
        SECTIONS[ArtifactSections]
    end
    
    subgraph "DAG Components"
        NODES[Nodes Array]
        EDGES[Edges Array]
        ENTRY[Entry Nodes]
        EXIT[Exit Nodes]
        META[DAG Metadata]
    end
    
    HEADER --> MAGIC
    HEADER --> VERSION
    HEADER --> TIMESTAMP
    HEADER --> HASH
    HEADER --> SIZE
    
    PAYLOAD --> METADATA
    PAYLOAD --> DAG
    PAYLOAD --> SECTIONS
    
    DAG --> NODES
    DAG --> EDGES
    DAG --> ENTRY
    DAG --> EXIT
    DAG --> META
    
    style HEADER fill:#fff4e1
    style PAYLOAD fill:#e8f5e9
    style DAG fill:#e1f5ff
```

## 9. LLM Registry Architecture

```mermaid
graph TB
    subgraph "LLMRegistry"
        REG[LLMRegistry]
        
        subgraph "Backend Providers"
            OPENAI[OpenAI Backend]
            ANTHROPIC[Anthropic Backend]
            GOOGLE[Google Backend]
            OLLAMA[Ollama Backend]
        end
        
        subgraph "Backend Interface"
            TRAIT[LLMBackend Trait]
            REQUEST[LLMRequest]
            RESPONSE[LLMResponse]
        end
        
        subgraph "Features"
            HEALTH[Health Checks]
            RETRY[Retry Logic]
            RESOLVER[Model Resolver]
        end
    end
    
    REG --> OPENAI
    REG --> ANTHROPIC
    REG --> GOOGLE
    REG --> OLLAMA
    
    OPENAI --> TRAIT
    ANTHROPIC --> TRAIT
    GOOGLE --> TRAIT
    OLLAMA --> TRAIT
    
    TRAIT --> REQUEST
    TRAIT --> RESPONSE
    
    REG --> HEALTH
    REG --> RETRY
    REG --> RESOLVER
    
    style REG fill:#e1f5ff
    style TRAIT fill:#fff4e1
```

## 10. Configuration Loading Flow

```mermaid
graph TD
    START[Driver Start] --> PROJECT[Search Project Config]
    
    PROJECT --> FIND[Walk up directories]
    FIND --> PROJ_FILE{.apxm/config.toml<br/>found?}
    
    PROJ_FILE -->|Yes| LOAD_PROJ[Load Project Config]
    PROJ_FILE -->|No| GLOBAL[Load Global Config]
    
    GLOBAL --> GLOBAL_FILE{~/.apxm/config.toml<br/>exists?}
    GLOBAL_FILE -->|Yes| LOAD_GLOBAL[Load Global Config]
    GLOBAL_FILE -->|No| DEFAULT[Use Default Config]
    
    LOAD_PROJ --> PARSE[Parse TOML]
    LOAD_GLOBAL --> PARSE
    DEFAULT --> PARSE
    
    PARSE --> APXM_CONFIG[ApXmConfig]
    APXM_CONFIG --> RUNTIME_CONFIG[RuntimeConfig]
    
    RUNTIME_CONFIG --> RUNTIME[Initialize Runtime]
    
    style START fill:#e8f5e9
    style APXM_CONFIG fill:#fff4e1
```

## 11. Error Propagation Flow

```mermaid
graph TD
    subgraph "Error Sources"
        COMPILER_ERR[Compiler Error]
        RUNTIME_ERR[Runtime Error]
        CONFIG_ERR[Config Error]
        IO_ERR[IO Error]
    end
    
    subgraph "Error Types"
        DRIVER_ERROR[DriverError]
        RUNTIME_ERROR[RuntimeError]
    end
    
    subgraph "Error Handling"
        PRINT[Print Error]
        SUGGEST[Suggestion System]
        EXIT[Exit Code]
    end
    
    COMPILER_ERR --> DRIVER_ERROR
    RUNTIME_ERR --> DRIVER_ERROR
    RUNTIME_ERR --> RUNTIME_ERROR
    CONFIG_ERR --> DRIVER_ERROR
    IO_ERR --> DRIVER_ERROR
    
    DRIVER_ERROR --> PRINT
    RUNTIME_ERROR --> PRINT
    
    PRINT --> SUGGEST
    SUGGEST --> EXIT
    
    style COMPILER_ERR fill:#ffebee
    style RUNTIME_ERR fill:#ffebee
    style EXIT fill:#e8f5e9
```

---

These diagrams provide visual representations of:
- System architecture and component relationships
- Data flow through compilation and execution
- Runtime execution details
- Memory and scheduler internals
- Configuration and error handling
- Shared operation definitions and end-to-end flow

Use these diagrams alongside the docs/architecture.md document for a complete understanding of the APXM system.
