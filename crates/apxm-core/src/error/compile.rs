//! Compilation errors.
//!
//! This module defines errors that occur during the compilation phase,
//! including parsing, type checking, verification, and optimization errors.
//!
//! All errors wrap [`Error`] to provide detailed, contextual error reporting
//! with source spans, suggestions, and help text.

use thiserror::Error as ThisError;

use crate::error::Error;

/// Errors that occur during compilation.
#[derive(Debug, ThisError)]
pub enum CompileError {
    /// Parse error with rich context
    #[error("Parse error: {0}")]
    Parse(Box<Error>),

    /// Type error with rich context
    #[error("Type error: {0}")]
    Type(Box<Error>),

    /// Verification error with rich context
    #[error("Verification error: {0}")]
    Verification(Box<Error>),

    /// Optimization error with rich context
    #[error("Optimization error: {0}")]
    Optimization(Box<Error>),

    /// Pass execution failed with rich context
    #[error("Pass execution failed: {0}")]
    PassFailed(Box<Error>),

    /// DAG construction error with rich context
    #[error("DAG construction failed: {0}")]
    DagConstruction(Box<Error>),

    /// Module not found (simple error, no rich context needed)
    #[error("Module not found: {name}")]
    ModuleNotFound {
        /// Name of the module that was not found.
        name: String,
    },
}

impl CompileError {
    /// Get the underlying Error if available
    pub fn as_error(&self) -> Option<&Error> {
        match self {
            CompileError::Parse(e) => Some(e),
            CompileError::Type(e) => Some(e),
            CompileError::Verification(e) => Some(e),
            CompileError::Optimization(e) => Some(e),
            CompileError::PassFailed(e) => Some(e),
            CompileError::DagConstruction(e) => Some(e),
            CompileError::ModuleNotFound { .. } => None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::codes::ErrorCode;
    use crate::error::span::Span;

    #[test]
    fn test_parse_error() {
        let span = Span::new("test.ais".to_string(), 5, 10, 1);
        let err = Error::new(
            ErrorCode::UnexpectedToken,
            "Unexpected token".to_string(),
            span,
        );
        let error = CompileError::Parse(Box::new(err));

        let display = format!("{}", error);
        assert!(display.contains("Parse error"));
        assert!(display.contains("E001"));
        assert!(display.contains("Unexpected token"));
    }

    #[test]
    fn test_type_error() {
        let span = Span::new("test.ais".to_string(), 10, 5, 3);
        let err = Error::new(
            ErrorCode::TypeMismatch,
            "expected 'String', got 'Number'".to_string(),
            span,
        );
        let error = CompileError::Type(Box::new(err));

        let display = format!("{}", error);
        assert!(display.contains("Type error"));
        assert!(display.contains("E101"));
        assert!(display.contains("expected 'String'"));
        assert!(display.contains("got 'Number'"));
    }

    #[test]
    fn test_type_error_with_note() {
        let span = Span::new("test.ais".to_string(), 10, 5, 3);
        let err = Error::new(
            ErrorCode::TypeMismatch,
            "expected 'String', got 'Number'".to_string(),
            span,
        )
        .with_note("Cannot convert number to string".to_string());
        let error = CompileError::Type(Box::new(err));

        let display = format!("{}", error);
        assert!(display.contains("Type error"));
        assert!(display.contains("E101"));
    }

    #[test]
    fn test_verification_error() {
        let span = Span::new("test.ais".to_string(), 15, 1, 5);
        let err = Error::new(
            ErrorCode::DagCycleDetected,
            "Cycle detected in DAG".to_string(),
            span,
        );
        let error = CompileError::Verification(Box::new(err));

        let display = format!("{}", error);
        assert!(display.contains("Verification error"));
        assert!(display.contains("E203"));
        assert!(display.contains("Cycle detected in DAG"));
    }

    #[test]
    fn test_optimization_error() {
        let span = Span::new("test.ais".to_string(), 20, 1, 10);
        let err = Error::new(
            ErrorCode::PassExecutionFailed,
            "Failed to apply optimization pass".to_string(),
            span,
        );
        let error = CompileError::Optimization(Box::new(err));

        let display = format!("{}", error);
        assert!(display.contains("Optimization error"));
        assert!(display.contains("E301"));
        assert!(display.contains("Failed to apply optimization pass"));
    }

    #[test]
    fn test_module_not_found_error() {
        let error = CompileError::ModuleNotFound {
            name: "my_module".to_string(),
        };

        let display = format!("{}", error);
        assert!(display.contains("Module not found"));
        assert!(display.contains("my_module"));
    }

    #[test]
    fn test_error_implements_std_error() {
        let span = Span::new("test.ais".to_string(), 1, 1, 1);
        let err = Error::new(ErrorCode::UnexpectedToken, "Test".to_string(), span);
        let error = CompileError::Parse(Box::new(err));

        // Verify it implements std::error::Error
        let _: &dyn std::error::Error = &error;
    }

    #[test]
    fn test_as_error() {
        let span = Span::new("test.ais".to_string(), 5, 10, 1);
        let err = Error::new(ErrorCode::UnexpectedToken, "Test error".to_string(), span);
        let error = CompileError::Parse(Box::new(err));

        assert!(error.as_error().is_some());
        let e = error.as_error().unwrap();
        assert_eq!(e.code, ErrorCode::UnexpectedToken);
    }

    #[test]
    fn test_pretty_print() {
        let span = Span::new("test.ais".to_string(), 1, 5, 3);
        let error = Error::new(
            ErrorCode::ExpectedExpression,
            "expected expression".to_string(),
            span,
        );
        let error = CompileError::Parse(Box::new(error));

        let source = "let x = ";
        let output = error.pretty_print(Some(source));
        assert!(output.contains("error[E002]"));
        assert!(output.contains("expected expression"));
    }
}
