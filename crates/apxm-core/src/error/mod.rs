//! Error handling utilities for APXM Core.
//!
//! This module centralizes all error types used across the core crate.
//! It provides compile-time, runtime, and security errors, as well as
//! shared error context helpers found in [`common`].

/// Shared error helpers such as [`SourceLocation`] and [`ErrorContext`].
pub mod common;
/// Errors produced during compilation phases such as parsing or verification.
pub mod compile;
/// Errors that can happen while executing operations at runtime.
pub mod runtime;
/// Policy, authorization, and rate-limiting errors.
pub mod security;

pub use common::{
    ErrorContext, ErrorContextExt, OpId, SourceLocation, TraceId, chain_errors, format_error,
};
pub use compile::CompileError;
pub use runtime::RuntimeError;
pub use security::SecurityError;
