//! APXM Core - Fundamental types and traits for the APXM system.
//!
//! This crate provides the foundational types, data structures, and traits
//! that all other APXM components depend on.

pub mod error;
pub mod logging;
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

pub use types::{
    AISOperation, AISOperationType, DependencyType, Edge, Node, NodeId, NodeMetadata, Number,
    Token, TokenId, TokenStatus, Value,
};
