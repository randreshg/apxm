//! APXM Core - Core types and traits for the APXM system.
//!
//! This crate provides basic types and data structures used by all APXM components.

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

pub use plan::{InnerPlanDsl, Plan, PlanStep};

pub use types::{
    AISOperation, AISOperationType, DependencyType, Edge, InstructionConfig, Node, NodeId,
    NodeMetadata, Number, Token, TokenId, TokenStatus, Value,
};
