//! Compilation errors.
//!
//! This module defines errors that occur during the compilation phase,
//! including parsing, type checking, verification, and optimization errors.

use thiserror::Error;

use crate::error::common::SourceLocation;

/// Errors that occur during compilation.
#[derive(Debug, Error)]
pub enum CompileError {
    /// Parse error during source code parsing.
    #[error("Parse error at {location}: {message}")]
    Parse {
        /// Source location where the parse error occurred.
        location: SourceLocation,
        /// Error message describing what went wrong.
        message: String,
    },

    /// Type error during type checking.
    #[error("Type error: expected {expected}, got {actual}{}", .message.as_ref().map(|m| format!(" - {}", m)).unwrap_or_default())]
    Type {
        /// Expected type.
        expected: String,
        /// Actual type that was found.
        actual: String,
        /// Optional additional message.
        message: Option<String>,
    },

    /// Verification error during DAG or program verification.
    #[error("Verification error: {message}")]
    Verification {
        /// Error message describing the verification failure.
        message: String,
    },

    /// Optimization error during optimization passes.
    #[error("Optimization error: {message}")]
    Optimization {
        /// Error message describing the optimization failure.
        message: String,
    },

    /// Module not found error.
    #[error("Module not found: {name}")]
    ModuleNotFound {
        /// Name of the moduel that was not found.
        name: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_error() {
        let location = SourceLocation::new("test.ais".to_string(), 5, 10);
        let error = CompileError::Parse {
            location: location.clone(),
            message: "Unexpected token".to_string(),
        };

        let display = format!("{}", error);
        assert!(display.contains("Parse error"));
        assert!(display.contains("test.ais:5:10"));
        assert!(display.contains("Unexpected token"));
    }

    #[test]
    fn test_type_error() {
        let error = CompileError::Type {
            expected: "String".to_string(),
            actual: "Number".to_string(),
            message: None,
        };

        let display = format!("{}", error);
        assert!(display.contains("Type error"));
        assert!(display.contains("expected String"));
        assert!(display.contains("got Number"));
    }

    #[test]
    fn test_type_error_with_message() {
        let error = CompileError::Type {
            expected: "String".to_string(),
            actual: "Number".to_string(),
            message: Some("Cannot convert number to string".to_string()),
        };

        let display = format!("{}", error);
        assert!(display.contains("Type error"));
        assert!(display.contains("expected String"));
        assert!(display.contains("got Number"));
        assert!(display.contains("Cannot convert number to string"));
    }

    #[test]
    fn test_verification_error() {
        let error = CompileError::Verification {
            message: "Cycle detected in DAG".to_string(),
        };

        let display = format!("{}", error);
        assert!(display.contains("Verification error"));
        assert!(display.contains("Cycle detected in DAG"));
    }

    #[test]
    fn test_optimization_error() {
        let error = CompileError::Optimization {
            message: "Failed to apply optimization pass".to_string(),
        };

        let display = format!("{}", error);
        assert!(display.contains("Optimization error"));
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
        let error = CompileError::Parse {
            location: SourceLocation::new("test.ais".to_string(), 1, 1),
            message: "Test".to_string(),
        };

        // Verify it implements std::error::Error
        let _: &dyn std::error::Error = &error;
    }
}
