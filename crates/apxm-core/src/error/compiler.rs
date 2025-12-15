//! Compiler error types
//!
//! Unified error handling for the MLIR compiler.
//!
//! All errors wrap [`Error`] to provide detailed, contextual error reporting.

use thiserror::Error as ThisError;

use crate::error::Error;

/// Result type for compiler operations
pub type Result<T> = std::result::Result<T, CompilerError>;

/// Compiler error variants
#[derive(ThisError, Debug)]
pub enum CompilerError {
    /// Failed to create compiler context
    #[error("Context creation failed: {0}")]
    ContextCreation(Box<Error>),

    /// Failed to perform context operation
    #[error("Context operation failed: {0}")]
    ContextOperation(Box<Error>),

    /// Failed to parse MLIR source
    #[error("Parse error: {0}")]
    Parse(Box<Error>),

    /// Module verification failed
    #[error("Verification error: {0}")]
    Verification(Box<Error>),

    /// Pass manager operation failed
    #[error("Pass manager error: {0}")]
    PassManager(Box<Error>),

    /// Pass execution failed
    #[error("Pass execution failed: {0}")]
    PassExecution(Box<Error>),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(Box<Error>),

    /// Invalid input provided to the compiler API
    #[error("Invalid input: {0}")]
    InvalidInput(Box<Error>),

    /// Code generation or compilation error
    #[error("Compilation error: {0}")]
    Compilation(Box<Error>),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(Box<Error>),

    /// Unsupported operation
    #[error("Unsupported operation: {0}")]
    Unsupported(Box<Error>),

    /// Internal compiler error (should not happen)
    #[error("Internal compiler error: {0}")]
    Internal(Box<Error>),
}

impl CompilerError {
    /// Get the underlying Error if available
    pub fn as_error(&self) -> Option<&Error> {
        match self {
            CompilerError::ContextCreation(e) => Some(e),
            CompilerError::ContextOperation(e) => Some(e),
            CompilerError::Parse(e) => Some(e),
            CompilerError::Verification(e) => Some(e),
            CompilerError::PassManager(e) => Some(e),
            CompilerError::PassExecution(e) => Some(e),
            CompilerError::Serialization(e) => Some(e),
            CompilerError::InvalidInput(e) => Some(e),
            CompilerError::Compilation(e) => Some(e),
            CompilerError::InvalidConfig(e) => Some(e),
            CompilerError::Unsupported(e) => Some(e),
            CompilerError::Internal(e) => Some(e),
            CompilerError::Io(_) => None,
            CompilerError::Json(_) => None,
        }
    }

    /// Pretty-print with source code
    pub fn pretty_print(&self, source: Option<&str>) -> String {
        if let Some(e) = self.as_error() {
            e.pretty_print(source)
        } else {
            format!("{}", self)
        }
    }
}

impl CompilerError {
    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::Parse(_) | Self::Verification(_) | Self::InvalidConfig(_) | Self::InvalidInput(_)
        )
    }

    /// Get error category for logging/metrics
    pub fn category(&self) -> &'static str {
        match self {
            Self::ContextCreation(_) | Self::ContextOperation(_) => "context",
            Self::Parse(_) => "parse",
            Self::Verification(_) => "verification",
            Self::PassManager(_) | Self::PassExecution(_) => "pass",
            Self::Serialization(_) => "serialization",
            Self::InvalidInput(_) => "invalid_input",
            Self::Compilation(_) => "compilation",
            Self::Io(_) => "io",
            Self::Json(_) => "json",
            Self::InvalidConfig(_) => "config",
            Self::Unsupported(_) => "unsupported",
            Self::Internal(_) => "internal",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::codes::ErrorCode;
    use crate::error::span::Span;

    #[test]
    fn test_error_categories() {
        let span = Span::new("test.mlir".to_string(), 1, 1, 1);
        let error = Error::new(ErrorCode::UnexpectedToken, "test".to_string(), span);
        let parse_err = CompilerError::Parse(Box::new(error));
        assert_eq!(parse_err.category(), "parse");
        assert!(parse_err.is_recoverable());

        let span2 = Span::new("test.mlir".to_string(), 1, 1, 1);
        let error2 = Error::new(ErrorCode::InternalError, "bug".to_string(), span2);
        let internal_err = CompilerError::Internal(Box::new(error2));
        assert_eq!(internal_err.category(), "internal");
        assert!(!internal_err.is_recoverable());
    }

    #[test]
    fn test_as_error() {
        let span = Span::new("test.mlir".to_string(), 5, 10, 1);
        let error = Error::new(
            ErrorCode::MLIRVerificationFailed,
            "Verification failed".to_string(),
            span,
        );
        let error = CompilerError::Verification(Box::new(error));

        assert!(error.as_error().is_some());
        let err = error.as_error().unwrap();
        assert_eq!(err.code, ErrorCode::MLIRVerificationFailed);
    }
}
