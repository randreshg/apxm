//! Core types for the APXM system.
//!
//! This module re-exports fundamental types from apxm-ais and adds runtime-specific types.
//! The single source of truth for AIS operations is in apxm-ais.

pub mod compiler;
pub mod execution;
pub mod identifiers;
pub mod intents;
pub mod models;
pub mod operations;
pub mod session;
pub mod values;

pub use compiler::{
    AgentBinary, CodegenOptions, CompilationStage, EmitFormat, OpDef, OpId, OpType,
    OptimizationLevel, PipelineConfig, stage_rank,
};
pub use execution::{
    DagMetadata, DependencyType, Edge, ExecutionDag, ExecutionStats, Node, NodeId, NodeMetadata,
    NodeStatus, OpStatus,
};
pub use identifiers::{
    CapabilityName, CheckpointId, ExecutionId, GoalIdType, MessageId, NodeIdType, OpIdType,
    SessionId, TokenIdType, TraceId,
};
pub use intents::{
    Entity, EntityType, ExportFormat, InspectTarget, Intent, MemoryQueryType, ProgramBuildStep,
};
pub use models::{FinishReason, LLMResponse, ModelCapabilities, ModelInfo, TokenUsage};

// Re-export from operations (which re-exports from apxm-ais)
pub use operations::metadata::{
    find_operation_by_name, get_operation_spec, OperationField, OperationSpec, ValidationError,
    AIS_OPERATIONS, INTERNAL_OPERATIONS,
};
pub use operations::{AISOperation, AISOperationType, OperationCategory, validate_operation};

pub use session::{Example, Message, MessageMetadata, MessageRole};
pub use values::{Number, Token, TokenId, TokenStatus, Value};
