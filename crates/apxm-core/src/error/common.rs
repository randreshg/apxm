//! Common error types and utilities
//!
//! This module provides shared types and utilties for error handling across APXM.

use std::collections::HashMap;
use std::fmt;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use crate::types::{AISOperationType, Value};

/// Operations correspond to node in the execution DAG, so OpID is the same as NodeId.
pub type OpId = u64;

/// Use for distributed tracing and error correlation.
pub type TraceId = String;

/// Represents a location in source code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceLocation {
    /// File path or identifier,
    pub file: String,
    /// Line number (1-indexed).
    pub line: usize,
    /// Column number (1-indexed).
    pub column: usize,
}

impl SourceLocation {
    /// Creates a new source location.
    pub fn new(file: String, line: usize, column: usize) -> Self {
        SourceLocation { file, line, column }
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

/// Error context for attaching additional information to errors.
///
/// This struct allow errors to carry contextyal information such as operation IDs,
/// trace IDs, and user information for better debugging and observability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ErrorContext {
    /// Operation identifier where the error occurred.
    pub operation_id: Option<OpId>,
    /// Type of operation that failed.
    pub operation_type: Option<AISOperationType>,
    /// Trace identifier for distributed tracing.
    pub trace_id: Option<TraceId>,
    /// Timestam when the error occurred.
    pub timestamp: SystemTime,
    /// Additional metadata as key-value pairs.
    pub metadata: HashMap<String, Value>,
}

impl ErrorContext {
    /// Creates a neow error context with the current timestamp.
    pub fn new() -> Self {
        ErrorContext {
            operation_id: None,
            operation_type: None,
            trace_id: None,
            timestamp: SystemTime::now(),
            metadata: HashMap::new(),
        }
    }

    /// Sets the operation ID.
    pub fn with_operation_id(mut self, op_id: OpId) -> Self {
        self.operation_id = Some(op_id);
        self
    }

    /// Sets the operation type.
    pub fn with_operation_type(mut self, op_type: AISOperationType) -> Self {
        self.operation_type = Some(op_type);
        self
    }

    /// Sets the trace ID.
    pub fn with_trace_id(mut self, trace_id: TraceId) -> Self {
        self.trace_id = Some(trace_id);
        self
    }

    /// Adds metadata.
    pub fn with_metadata(mut self, key: String, value: Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait for adding context to errors.
///
/// This trait allows any error type to attach an `ErrorContext` for better observability
pub trait ErrorContextExt {
    /// Attaches context to this error.
    fn with_context(self, context: ErrorContext) -> Self;

    /// Gets the context from this error, if any.
    fn context(&self) -> Option<&ErrorContext>;
}

/// Chains multiple errors into a formatted string.
///
/// This function formats a chain of errors, showing the error hierarchy
/// from the root cause to the most recent error.
///
/// # Arguments
///
/// * `errors` - A vector of boxed errors to chain together
///
/// # Returns
///
/// A formatted string representing the error chain.
pub fn chain_errors(errors: Vec<Box<dyn std::error::Error>>) -> String {
    if let Some(result) = errors
        .iter()
        .map(|e| e.to_string())
        .reduce(|acc, e| format!("{}\n  caused by: {}", acc, e))
    {
        result
    } else {
        String::from("No errors")
    }
}

/// Formats a single error with its context.
///
/// This function formats an error along with any attached context information,
/// providing a error message debugging.
///
/// # Arguments
///
/// * `errors` - The error to format
///
/// # Returns
///
/// A formated string with error message and context.
pub fn format_error(error: &dyn std::error::Error) -> String {
    std::iter::successors(Some(error), |e| e.source())
        .enumerate()
        .map(|(i, err)| match i {
            0 => err.to_string(),
            1 => format!("\n    caused by:\n    {}", err),
            _ => format!("\n    {}", err),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_location_new() {
        let loc = SourceLocation::new("test.ais".to_string(), 10, 5);
        assert_eq!(loc.file, "test.ais");
        assert_eq!(loc.line, 10);
        assert_eq!(loc.column, 5);
    }

    #[test]
    fn test_source_location_display() {
        let loc = SourceLocation::new("test.ais".to_string(), 10, 5);
        assert_eq!(loc.to_string(), "test.ais:10:5");
    }

    #[test]
    fn test_error_context_new() {
        let ctx = ErrorContext::new();
        assert!(ctx.operation_id.is_none());
        assert!(ctx.operation_type.is_none());
        assert!(ctx.trace_id.is_none());
    }

    #[test]
    fn test_error_context_builder() {
        let ctx = ErrorContext::new()
            .with_operation_id(42)
            .with_operation_type(AISOperationType::Inv)
            .with_trace_id("trace-123".to_string())
            .with_metadata("key".to_string(), Value::String("value".to_string()));

        assert_eq!(ctx.operation_id, Some(42));
        assert_eq!(ctx.operation_type, Some(AISOperationType::Inv));
        assert_eq!(ctx.trace_id, Some("trace-123".to_string()));
        assert_eq!(
            ctx.metadata.get("key"),
            Some(&Value::String("value".to_string()))
        );
    }

    #[test]
    fn test_chain_errors_empty() {
        let errors: Vec<Box<dyn std::error::Error>> = vec![];
        assert_eq!(chain_errors(errors), "No errors");
    }

    #[test]
    fn test_chain_errors_single() {
        use std::io;
        let errors: Vec<Box<dyn std::error::Error>> = vec![Box::new(io::Error::new(
            io::ErrorKind::NotFound,
            "File not found",
        ))];
        let result = chain_errors(errors);
        assert!(result.contains("File not found"));
    }

    #[test]
    fn test_format_error() {
        use std::io;
        let err: Box<dyn std::error::Error> =
            Box::new(io::Error::new(io::ErrorKind::NotFound, "File not found"));
        let result = format_error(err.as_ref());
        assert!(result.contains("File not found"));
    }
}
