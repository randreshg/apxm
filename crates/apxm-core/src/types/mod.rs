//! Core type definitions for the APXM system.
//!
//! Covers execution graph primitives (`execution`), value representations
//! (`values`), compiler pipeline options (`compiler`), LLM model metadata
//! (`models`), session/message types (`session`), provider specifications
//! (`provider_spec`), typed identifiers (`identifiers`), intent routing
//! (`intents`), and instruction configuration (`config`).
//!
//! AIS operation types are re-exported from `apxm-ais` via `operations`.

pub mod compiler;
pub mod config;
pub mod execution;
pub mod identifiers;
pub mod intents;
pub mod models;
pub mod operations;
pub mod provider_spec;
pub mod session;
pub mod values;

pub use compiler::{
    CodegenOptions, CompilationStage, EmitFormat, OptimizationLevel, PipelineConfig, stage_rank,
};
pub use execution::{
    Agent, AgentFlow, AgentId, AgentMetadata, CapabilityDeclaration, Codelet, CodeletDag,
    CodeletId, CodeletMetadata, DagMetadata, DependencyType, Edge, ExecutionDag, ExecutionStats,
    MemoryDeclaration, Node, NodeId, NodeMetadata, NodeStatus, OpStatus,
};
pub use identifiers::{
    CapabilityName, CheckpointId, ExecutionId, GoalIdType, MessageId, NodeIdType, OpIdType,
    SessionId, TokenIdType, TraceId,
};
pub use intents::{
    Entity, EntityType, ExportFormat, InspectTarget, Intent, MemoryQueryType, ProgramBuildStep,
};
pub use models::{
    FinishReason, LLMResponse, ModelCapabilities, ModelInfo, TokenUsage, ToolCall, ToolResult,
};

// Re-export from operations (which re-exports from apxm-ais)
pub use operations::metadata::{
    AIS_OPERATIONS, OperationField, OperationSpec, ValidationError, get_operation_spec,
};
pub use operations::{AISOperation, AISOperationType, OperationCategory, validate_operation};

pub use session::{Example, Message, MessageMetadata, MessageRole};
pub use values::{Number, Token, TokenId, TokenStatus, Value};

pub use config::InstructionConfig;
pub use provider_spec::{
    BUILTIN_PROVIDERS, BuiltinProviderSpec, ProviderProtocol, ProviderSpec,
    resolve_builtin_provider, resolve_provider_spec,
};
