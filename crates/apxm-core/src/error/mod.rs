//! Error handling utilities for APXM Core.
//!
//! This module centralizes all error types used across the core crate.
//! It provides compile-time, runtime, and security errors, as well as
//! shared error context helpers found in [`common`].
//!
//! # Rich Error System
//!
//! The error system is built around [`Error`], which provides:
//! - Error codes (E001, E002, etc.)
//! - Source spans with snippets
//! - Actionable suggestions
//! - Educational help text
//! - Error context (operation_id, trace_id, etc.)
//!
//! All error types (`CompileError`, `CompilerError`, `RuntimeError`) wrap
//! `Error` to provide consistent, detailed error reporting.

/// Error type with full context (like Rust compiler errors).
pub mod api;
/// Error builders for common error patterns.
pub mod builder;
/// Errors related to the CLI interface.
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
