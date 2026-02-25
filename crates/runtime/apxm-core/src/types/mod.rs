//! Core types for the APXM system.
//!
//! This module re-exports fundamental types from apxm-ais and adds runtime-specific types.
//! The single source of truth for AIS operations is in apxm-ais.

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
    AgentBinary, CodegenOptions, CompilationStage, EmitFormat, OpDef, OpId, OpType,
    OptimizationLevel, PipelineConfig, stage_rank,
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
    AIS_OPERATIONS, INTERNAL_OPERATIONS, OperationField, OperationSpec, ValidationError,
    find_operation_by_name, get_operation_spec,
};
pub use operations::{AISOperation, AISOperationType, OperationCategory, validate_operation};

pub use session::{Example, Message, MessageMetadata, MessageRole};
pub use values::{Number, Token, TokenId, TokenStatus, Value};

pub use config::InstructionConfig;
pub use provider_spec::{
    BUILTIN_PROVIDERS, BuiltinProviderSpec, ProviderProtocol, ProviderSpec,
    resolve_builtin_provider, resolve_provider_spec,
};
