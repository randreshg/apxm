//! Shared core crate providing canonical types, error definitions, and constants
//! used across both the APXM compiler and runtime.
//!
//! - **`types`** -- Execution graph primitives (`Node`, `Edge`, `ExecutionDag`),
//!   value representations (`Value`, `Token`, `Number`), compiler options,
//!   session/message models, and provider specifications.
//! - **`error`** -- Structured error types (`RuntimeError`, `CompilerError`,
//!   `CompileError`, `CliError`, `SecurityError`) with error codes, source
//!   locations, and diagnostic suggestions.
//! - **`constants`** -- Centralised string keys for graph attributes, inner-plan
//!   payloads, and diagnostic modes so all front-ends and back-ends stay in sync.

pub mod constants;
pub mod error;
pub mod logging;
pub mod paths;
pub mod plan;
pub mod types;
pub mod utils;

pub use error::{
    cli::{CliError, CliResult},
    common::{ErrorContext, ErrorContextExt, OpId, SourceLocation, TraceId},
    compile::CompileError,
    compiler::CompilerError,
    runtime::RuntimeError,
    security::SecurityError,
};

pub use plan::{InnerPlanPayload, Plan, PlanStep};

pub use types::{
    AISOperation, AISOperationType, DependencyType, Edge, InstructionConfig, Node, NodeId,
    NodeMetadata, Number, Token, TokenId, TokenStatus, Value,
};
