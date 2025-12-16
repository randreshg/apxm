//! Core types for the APXM system.
//!
//! This module contains data structures used throughout APXM:
//! - `Number`: Numeric values
//! - `Value`: Unified values
//! - `Token`: Token-based dataflow
//! - `AISOperationType`: Operation types
//! - `Node`, `Edge`, `ExecutionDAG`: Execution graph
//! - `Message`, `MessageRole`: Conversation types
//! - `Intent`, `Entity`: Intent classification

pub mod compiler;
pub mod execution;
pub mod identifiers;
pub mod intents;
pub mod operations;
pub mod session;
pub mod values;

pub use compiler::{
    AgentBinary, CodegenOptions, CompilationStage, EmitFormat, OpDef, OpId, OpType,
    OptimizationLevel, PipelineConfig, stage_rank,
};
pub use execution::{DagMetadata, DependencyType, Edge, ExecutionDag, Node, NodeId, NodeMetadata};
pub use identifiers::{
    CapabilityName, CheckpointId, ExecutionId, GoalIdType, MessageId, NodeIdType, OpIdType,
    SessionId, TokenIdType, TraceId,
};
pub use intents::{
    Entity, EntityType, ExportFormat, InspectTarget, Intent, MemoryQueryType, ProgramBuildStep,
};
pub use operations::metadata::operations as operation_definitions;
pub use operations::metadata::{OperationField, OperationMetadata, find_operation};
pub use operations::{AISOperation, AISOperationType, validate_operation};
pub use session::{Example, Message, MessageMetadata, MessageRole};
pub use values::{Number, Token, TokenId, TokenStatus, Value};
