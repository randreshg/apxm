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
    fn test_as_error() -> Result<(), Box<dyn std::error::Error>> {
        let span = Span::new("test.ais".to_string(), 5, 10, 1);
        let err = Error::new(ErrorCode::UnexpectedToken, "Test error".to_string(), span);
        let error = CompileError::Parse(Box::new(err));

        match error.as_error() {
            Some(e) => assert_eq!(e.code, ErrorCode::UnexpectedToken),
            None => return Err("expected underlying Error in CompileError::Parse".into()),
        }

        Ok(())
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
