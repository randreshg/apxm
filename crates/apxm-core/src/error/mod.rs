//! Error handling for APXM Core.
//!
//! This module contains all error types used in the core crate.
//! Error types include compile-time, runtime, and security errors.
//!
//! # Error System
//!
//! The system uses [`Error`] with:
//! - Error codes (E001, E002, etc.)
//! - Source location information
//! - Error suggestions
//! - Context information
//!
//! All specific error types wrap `Error` for consistency.

/// Error type with full context (like Rust compiler errors).
pub mod api;
/// Error builders for common error patterns.
pub mod builder;
/// Errors related to tooling interfaces.
pub mod cli;
/// Error codes for all APXM components.
pub mod codes;
/// Shared error helpers such as [`SourceLocation`] and [`ErrorContext`].
pub mod common;
/// Errors produced during compilation phases such as parsing or verification.
pub mod compile;
/// Errors related to the MLIR compiler infrastructure.
pub mod compiler;
/// Errors that can happen while executing operations at runtime.
pub mod runtime;
/// Policy, authorization, and rate-limiting errors.
pub mod security;
/// Source code spans for error reporting.
pub mod span;
/// Suggestions for fixing errors.
pub mod suggestion;

pub use api::Error;
pub use builder::ErrorBuilder;
pub use cli::{CliError, CliResult};
pub use codes::ErrorCode;
pub use common::{
    ErrorContext, ErrorContextExt, OpId, SourceLocation, TraceId, chain_errors, format_error,
};
pub use compile::CompileError;
pub use compiler::CompilerError;
pub use runtime::RuntimeError;
pub use security::SecurityError;
pub use span::Span;
pub use suggestion::Suggestion;
