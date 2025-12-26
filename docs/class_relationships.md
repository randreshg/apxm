# APXM Class Relationships

This document provides detailed class relationship diagrams for understanding the object-oriented structure of APXM.

## 1. Linker Package Classes

```mermaid
classDiagram
    class Linker {
        -Compiler compiler
        -RuntimeExecutor runtime
        +new(config: LinkerConfig) Result~Linker, DriverError~
        +compile_only(input: Path, mlir: bool) Result~Module, DriverError~
        +run(input: Path, mlir: bool) Result~LinkResult, DriverError~
        +runtime_llm_registry() Arc~LLMRegistry~
        +runtime_capabilities() Vec~String~
    }
    
    class Compiler {
        +compile(input: Path, mlir: bool) Result~Module, CompilerError~
    }
    
    class RuntimeExecutor {
        -Runtime runtime
        +new(config: &LinkerConfig) Result~RuntimeExecutor, DriverError~
        +execute(dag: ExecutionDag) Result~RuntimeExecutionResult, DriverError~
        +llm_registry() Arc~LLMRegistry~
        +capability_names() Vec~String~
        +capability_system() &CapabilitySystem
    }
    
    class LinkerConfig {
        +runtime_config: RuntimeConfig
        +apxm_config: ApXmConfig
        +from_apxm_config(config: ApXmConfig) LinkerConfig
    }
    
    class LinkResult {
        +module: Module
        +artifact: Artifact
        +execution: RuntimeExecutionResult
    }
    
    Linker --> Compiler : uses
    Linker --> RuntimeExecutor : uses
    Linker --> LinkResult : returns
    RuntimeExecutor --> Runtime : contains
    LinkerConfig --> RuntimeConfig : contains
    LinkerConfig --> ApXmConfig : contains
```

## 2. Runtime Package Classes

```mermaid
classDiagram
    class Runtime {
        -RuntimeConfig config
        -Arc~MemorySystem~ memory
        -Arc~LLMRegistry~ llm_registry
        -Arc~CapabilitySystem~ capability_system
        -Aam aam
        -DataflowScheduler scheduler
        -Arc~dyn InnerPlanLinker~ inner_plan_linker
        +new(config: RuntimeConfig) Result~Runtime, RuntimeError~
        +execute(dag: ExecutionDag) Result~RuntimeExecutionResult, RuntimeError~
        +execute_artifact(artifact: Artifact) Result~RuntimeExecutionResult, RuntimeError~
        +execute_artifact_bytes(bytes: &[u8]) Result~RuntimeExecutionResult, RuntimeError~
        +set_inner_plan_linker(linker: Arc~dyn InnerPlanLinker~)
        +memory() &MemorySystem
        +llm_registry() &LLMRegistry
        +capability_system() &CapabilitySystem
        +aam() &Aam
    }
    
    class RuntimeConfig {
        +memory_config: MemoryConfig
        +scheduler_config: SchedulerConfig
        +in_memory() RuntimeConfig
        +with_scheduler_config(config: SchedulerConfig) RuntimeConfig
    }
    
    class MemorySystem {
        +read(space: MemorySpace, key: &str) Result~Option~Value~, RuntimeError~
        +write(space: MemorySpace, key: String, value: Value) Result~(), RuntimeError~
        +stm() &STM
        +ltm() &LTM
    }
    
    class DataflowScheduler {
        -SchedulerConfig config
        +execute(dag: ExecutionDag, executor: Arc~ExecutorEngine~, context: ExecutionContext) Result~(HashMap~u64, Value~, ExecutionStats), RuntimeError~
    }
    
    class ExecutorEngine {
        -ExecutionContext context
        +new(context: ExecutionContext) ExecutorEngine
        +execute_dag(dag: ExecutionDag) Result~ExecutionResult, RuntimeError~
        +execute_node(node: &Node, inputs: Vec~Value~) Result~Value, RuntimeError~
    }
    
    class ExecutionContext {
        +memory: Arc~MemorySystem~
        +llm_registry: Arc~LLMRegistry~
        +capability_system: Arc~CapabilitySystem~
        +aam: Aam
        +inner_plan_linker: Arc~dyn InnerPlanLinker~
        +execution_id: Uuid
        +new(memory, llm_registry, capability_system, aam) ExecutionContext
    }
    
    class OperationDispatcher {
        +dispatch(context: &ExecutionContext, node: &Node, inputs: Vec~Value~) Result~Value, RuntimeError~
    }
    
    Runtime --> RuntimeConfig : uses
    Runtime --> MemorySystem : contains
    Runtime --> DataflowScheduler : contains
    Runtime --> ExecutorEngine : creates
    ExecutorEngine --> ExecutionContext : uses
    ExecutorEngine --> OperationDispatcher : uses
    DataflowScheduler --> ExecutorEngine : uses
    ExecutionContext --> MemorySystem : references
    ExecutionContext --> LLMRegistry : references
    ExecutionContext --> CapabilitySystem : references
```

## 3. Core Types Classes

```mermaid
classDiagram
    class ExecutionDag {
        +nodes: Vec~Node~
        +edges: Vec~Edge~
        +entry_nodes: Vec~u64~
        +exit_nodes: Vec~u64~
        +metadata: DagMetadata
        +validate() Result~(), String~
    }
    
    class Node {
        +id: u64
        +op_type: AISOperationType
        +attributes: HashMap~String, Value~
        +input_tokens: Vec~u64~
        +output_tokens: Vec~u64~
        +metadata: NodeMetadata
    }
    
    class Edge {
        +from_node: u64
        +to_node: u64
        +from_token: u64
        +to_token: u64
    }
    
    class AISOperationType {
        <<enumeration>>
        ConstStr
        UMem
        QMem
        Invoke
        Communicate
        Plan
        Reflect
        Branch
        LoopStart
        LoopEnd
        Merge
        WaitAll
        TryCatch
        Return
        ...
    }
    
    class Value {
        <<enumeration>>
        String(String)
        Number(Number)
        Boolean(bool)
        Null
        Array(Vec~Value~)
        Object(HashMap~String, Value~)
        +to_json() Result~JsonValue~
        +to_string() String
    }
    
    class Artifact {
        -metadata: ArtifactMetadata
        -dag: ExecutionDag
        -sections: Vec~ArtifactSection~
        -flags: u32
        +new(metadata: ArtifactMetadata, dag: ExecutionDag) Artifact
        +dag() &ExecutionDag
        +into_dag() ExecutionDag
        +metadata() &ArtifactMetadata
        +to_bytes() Result~Vec~u8~, ArtifactError~
        +from_bytes(bytes: &[u8]) Result~Artifact, ArtifactError~
        +payload_hash() Result~[u8; 32], ArtifactError~
    }
    
    class ArtifactMetadata {
        +module_name: Option~String~
        +created_at: u64
        +compiler_version: String
    }
    
    ExecutionDag --> Node : contains
    ExecutionDag --> Edge : contains
    Node --> AISOperationType : uses
    Node --> Value : uses
    Artifact --> ExecutionDag : contains
    Artifact --> ArtifactMetadata : contains
```

## 4. Memory System Classes

```mermaid
classDiagram
    class MemorySystem {
        -Arc~STM~ stm
        -Arc~LTM~ ltm
        -EpisodicMemory episodic
        +new(config: MemoryConfig) Result~MemorySystem, RuntimeError~
        +read(space: MemorySpace, key: &str) Result~Option~Value~, RuntimeError~
        +write(space: MemorySpace, key: String, value: Value) Result~(), RuntimeError~
        +stm() &STM
        +ltm() &LTM
        +len(space: MemorySpace) Result~usize, RuntimeError~
    }
    
    class MemoryConfig {
        +stm_backend: MemoryBackend
        +ltm_backend: MemoryBackend
        +in_memory_ltm() MemoryConfig
    }
    
    class STM {
        +read(key: &str) Result~Option~Value~, RuntimeError~
        +write(key: String, value: Value) Result~(), RuntimeError~
        +len() Result~usize, RuntimeError~
    }
    
    class LTM {
        +read(key: &str) Result~Option~Value~, RuntimeError~
        +write(key: String, value: Value) Result~(), RuntimeError~
        +len() Result~usize, RuntimeError~
    }
    
    class MemoryBackend {
        <<trait>>
        +read(key: &str) Result~Option~Value~, RuntimeError~
        +write(key: String, value: Value) Result~(), RuntimeError~
        +len() Result~usize, RuntimeError~
    }
    
    class InMemoryBackend {
        -HashMap~String, Value~ data
        +read(key: &str) Result~Option~Value~, RuntimeError~
        +write(key: String, value: Value) Result~(), RuntimeError~
    }
    
    class SqliteBackend {
        -Connection db
        +read(key: &str) Result~Option~Value~, RuntimeError~
        +write(key: String, value: Value) Result~(), RuntimeError~
    }
    
    MemorySystem --> STM : contains
    MemorySystem --> LTM : contains
    MemorySystem --> MemoryConfig : uses
    STM --> MemoryBackend : uses
    LTM --> MemoryBackend : uses
    MemoryBackend <|.. InMemoryBackend : implements
    MemoryBackend <|.. SqliteBackend : implements
```

## 5. LLM Registry Classes

```mermaid
classDiagram
    class LLMRegistry {
        -HashMap~String, Arc~dyn LLMBackend~~ backends
        -String default_backend
        +new() LLMRegistry
        +register(name: String, backend: Arc~dyn LLMBackend~)
        +generate_with_backend(name: &str, request: LLMRequest) Result~LLMResponse, RegistryError~
        +generate(request: LLMRequest) Result~LLMResponse, RegistryError~
        +list_backends() Vec~String~
    }
    
    class LLMBackend {
        <<trait>>
        +generate(request: LLMRequest) Result~LLMResponse, BackendError~
        +health_check() Result~(), BackendError~
    }
    
    class LLMRequest {
        +prompt: String
        +max_tokens: Option~usize~
        +temperature: Option~f64~
        +with_max_tokens(tokens: usize) LLMRequest
        +with_temperature(temp: f64) LLMRequest
    }
    
    class LLMResponse {
        +content: String
        +model: String
        +usage: Option~Usage~
    }
    
    class OpenAIBackend {
        -String api_key
        -String model
        -String endpoint
        +generate(request: LLMRequest) Result~LLMResponse, BackendError~
    }
    
    class AnthropicBackend {
        -String api_key
        -String model
        +generate(request: LLMRequest) Result~LLMResponse, BackendError~
    }
    
    class GoogleBackend {
        -String api_key
        -String model
        +generate(request: LLMRequest) Result~LLMResponse, BackendError~
    }
    
    class OllamaBackend {
        -String endpoint
        -String model
        +generate(request: LLMRequest) Result~LLMResponse, BackendError~
    }
    
    LLMRegistry --> LLMBackend : manages
    LLMBackend <|.. OpenAIBackend : implements
    LLMBackend <|.. AnthropicBackend : implements
    LLMBackend <|.. GoogleBackend : implements
    LLMBackend <|.. OllamaBackend : implements
    LLMBackend --> LLMRequest : uses
    LLMBackend --> LLMResponse : returns
```

## 6. Capability System Classes

```mermaid
classDiagram
    class CapabilitySystem {
        -HashMap~String, CapabilityMetadata~ capabilities
        -Aam aam
        +new() CapabilitySystem
        +with_aam(aam: Aam) CapabilitySystem
        +register(name: String, metadata: CapabilityMetadata)
        +execute(name: &str, args: HashMap~String, Value~) Result~Value, RuntimeError~
        +list_capability_names() Vec~String~
        +get_metadata(name: &str) Option~&CapabilityMetadata~
    }
    
    class CapabilityMetadata {
        +name: String
        +description: String
        +parameters: Vec~Parameter~
        +executor: CapabilityExecutor
    }
    
    class CapabilityExecutor {
        <<trait>>
        +execute(args: HashMap~String, Value~) Result~Value, RuntimeError~
    }
    
    class Parameter {
        +name: String
        +type: ParameterType
        +required: bool
        +description: Option~String~
    }
    
    CapabilitySystem --> CapabilityMetadata : manages
    CapabilityMetadata --> CapabilityExecutor : uses
    CapabilityMetadata --> Parameter : contains
```

## 7. Scheduler Classes

```mermaid
classDiagram
    class DataflowScheduler {
        -SchedulerConfig config
        -ReadySet ready_set
        -PriorityQueue queue
        -ConcurrencyControl concurrency
        +new(config: SchedulerConfig) DataflowScheduler
        +execute(dag: ExecutionDag, executor: Arc~ExecutorEngine~, context: ExecutionContext) Result~(HashMap~u64, Value~, ExecutionStats), RuntimeError~
    }
    
    class ReadySet {
        -HashSet~u64~ ready_nodes
        -HashMap~u64, usize~ dependency_counts
        +add_node(node_id: u64)
        +mark_ready(node_id: u64)
        +is_ready(node_id: u64) bool
        +get_ready() Vec~u64~
    }
    
    class PriorityQueue {
        -Vec~Vec~Node~~ queues
        +enqueue(node: Node, priority: Priority)
        +dequeue() Option~Node~
        +is_empty() bool
    }
    
    class Priority {
        <<enumeration>>
        Critical
        High
        Normal
        Low
    }
    
    class ConcurrencyControl {
        -Semaphore semaphore
        +acquire() Result~(), RuntimeError~
        +release()
    }
    
    class SchedulerConfig {
        +max_concurrency: usize
        +retry_attempts: usize
        +deadlock_timeout_ms: u64
    }
    
    DataflowScheduler --> ReadySet : uses
    DataflowScheduler --> PriorityQueue : uses
    DataflowScheduler --> ConcurrencyControl : uses
    DataflowScheduler --> SchedulerConfig : uses
    PriorityQueue --> Priority : uses
```

## 8. Operation Handler Classes

```mermaid
classDiagram
    class OperationDispatcher {
        +dispatch(context: &ExecutionContext, node: &Node, inputs: Vec~Value~) Result~Value, RuntimeError~
    }
    
    class OperationHandler {
        <<trait>>
        +handle(context: &ExecutionContext, node: &Node, inputs: Vec~Value~) Result~Value, RuntimeError~
    }
    
    class ConstStrHandler {
        +handle(context, node, inputs) Result~Value, RuntimeError~
    }
    
    class UMemHandler {
        +handle(context, node, inputs) Result~Value, RuntimeError~
    }
    
    class QMemHandler {
        +handle(context, node, inputs) Result~Value, RuntimeError~
    }
    
    class InvokeHandler {
        +handle(context, node, inputs) Result~Value, RuntimeError~
    }
    
    class CommunicateHandler {
        +handle(context, node, inputs) Result~Value, RuntimeError~
    }
    
    class PlanHandler {
        +handle(context, node, inputs) Result~Value, RuntimeError~
    }
    
    class ReflectHandler {
        +handle(context, node, inputs) Result~Value, RuntimeError~
    }
    
    class BranchHandler {
        +handle(context, node, inputs) Result~Value, RuntimeError~
    }
    
    OperationDispatcher --> OperationHandler : uses
    OperationHandler <|.. ConstStrHandler : implements
    OperationHandler <|.. UMemHandler : implements
    OperationHandler <|.. QMemHandler : implements
    OperationHandler <|.. InvokeHandler : implements
    OperationHandler <|.. CommunicateHandler : implements
    OperationHandler <|.. PlanHandler : implements
    OperationHandler <|.. ReflectHandler : implements
    OperationHandler <|.. BranchHandler : implements
```

## 9. Error Hierarchy

```mermaid
classDiagram
    class Error {
        <<trait>>
        +source() Option~&dyn Error~
        +description() String
    }
    
    class DriverError {
        <<enumeration>>
        Compiler(CompilerError)
        Runtime(RuntimeError)
        Config { message: String }
        Io(io::Error)
    }
    
    class RuntimeError {
        <<enumeration>>
        State(String)
        Execution(String)
        Memory(String)
        Capability(String)
    }
    
    class CompilerError {
        +message: String
        +span: Option~Span~
        +suggestion: Option~Suggestion~
    }
    
    Error <|.. DriverError
    Error <|.. RuntimeError
    Error <|.. CompilerError
    DriverError --> CompilerError : contains
    DriverError --> RuntimeError : contains
```

---

## Key Relationships Summary

1. **Composition**: `Runtime` contains `MemorySystem`, `LLMRegistry`, `CapabilitySystem`, `DataflowScheduler`
2. **Aggregation**: `Linker` uses `Compiler` and `RuntimeExecutor` but doesn't own them
3. **Dependency**: `ExecutorEngine` depends on `ExecutionContext` and `OperationDispatcher`
4. **Trait Implementation**: Multiple backends implement `LLMBackend` trait
5. **Strategy Pattern**: Different operation handlers implement `OperationHandler` trait
6. **Factory Pattern**: `LLMRegistry` creates and manages backend instances
7. **Observer Pattern**: Runtime emits tracing/metrics events

These relationships enable:
- **Modularity**: Components can be swapped or extended independently
- **Testability**: Dependencies can be mocked or stubbed
- **Flexibility**: New operation types, backends, and capabilities can be added
- **Maintainability**: Clear separation of concerns
