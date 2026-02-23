//! Convenient error builders for common error patterns.
//!
//! This module provides centralized error construction helpers to eliminate
//! duplication across the codebase. All builders create `Error` instances
//! with appropriate error codes and default spans.

use crate::error::{Error, ErrorCode};

/// Builder for errors without source location information.
///
/// This builder is used when creating errors from contexts where source
/// location is not available (e.g., FFI boundaries, runtime errors).
pub struct ErrorBuilder;

impl ErrorBuilder {
    /// Create a generic error with unknown span.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let err = ErrorBuilder::generic(ErrorCode::InternalError, "Something went wrong");
    /// ```
    pub fn generic(code: ErrorCode, message: impl Into<String>) -> Error {
        Error::new_generic(code, message)
    }

    /// Create an internal error with unknown span.
    pub fn internal(message: impl Into<String>) -> Error {
        Self::generic(ErrorCode::InternalError, message)
    }

    /// Create a parse error with unknown span.
    ///
    /// Uses `ErrorCode::InternalError` as the default code for parse errors
    /// without specific location information.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let err = ErrorBuilder::parse("Failed to parse input");
    /// ```
    pub fn parse(message: impl Into<String>) -> Error {
        Self::generic(ErrorCode::InternalError, message)
    }

    /// Create a verification error with unknown span.
    ///
    /// Uses `ErrorCode::MLIRVerificationFailed` for MLIR verification failures.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let err = ErrorBuilder::verification("Module verification failed");
    /// ```
    pub fn verification(message: impl Into<String>) -> Error {
        Error::new_generic(ErrorCode::MLIRVerificationFailed, message)
    }

    /// Create a pass execution error with unknown span.
    ///
    /// Uses `ErrorCode::PassExecutionFailed` for optimization pass failures.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let err = ErrorBuilder::pass_execution("Normalize pass failed");
    /// ```
    pub fn pass_execution(message: impl Into<String>) -> Error {
        Error::new_generic(ErrorCode::PassExecutionFailed, message)
    }

    /// Create a pass manager error with unknown span.
    ///
    /// Allows specifying a custom error code for different pass manager failures.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let err = ErrorBuilder::pass_manager(
    ///     ErrorCode::PassNotFound,
    ///     "Pass 'normalize' not found"
    /// );
    /// ```
    pub fn pass_manager(code: ErrorCode, message: impl Into<String>) -> Error {
        Error::new_generic(code, message)
    }

    /// Create a serialization error with unknown span.
    ///
    /// Uses `ErrorCode::InternalError` for serialization failures.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let err = ErrorBuilder::serialization("Failed to serialize module");
    /// ```
    pub fn serialization(message: impl Into<String>) -> Error {
        Error::new_generic(ErrorCode::InternalError, message)
    }

    /// Create a context creation error with unknown span.
    ///
    /// Uses `ErrorCode::InternalError` for compiler context creation failures.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let err = ErrorBuilder::context_creation("Failed to initialize MLIR context");
    /// ```
    pub fn context_creation(message: impl Into<String>) -> Error {
        Error::new_generic(ErrorCode::InternalError, message)
    }

    /// Create a context operation error with unknown span.
    ///
    /// Uses `ErrorCode::InternalError` for compiler context operation failures.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let err = ErrorBuilder::context_operation("Failed to register dialect");
    /// ```
    pub fn context_operation(message: impl Into<String>) -> Error {
        Error::new_generic(ErrorCode::InternalError, message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generic_builder() {
        let err = ErrorBuilder::generic(ErrorCode::TypeMismatch, "Type error");
        assert_eq!(err.code, ErrorCode::TypeMismatch);
        assert_eq!(err.message, "Type error");
        assert_eq!(err.primary_span.file, "<unknown>");
    }

    #[test]
    fn test_context_operation_builder() {
        let err = ErrorBuilder::context_operation("Context operation failed");
        assert_eq!(err.code, ErrorCode::InternalError);
        assert_eq!(err.message, "Context operation failed");
    }
}
