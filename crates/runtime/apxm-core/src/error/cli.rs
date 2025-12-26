//! Tooling error types.
//!
//! Provides command-level error handling with proper exit codes
//! and user-friendly error messages.

use crate::error::{ErrorCode, Suggestion};
use std::path::PathBuf;
use thiserror::Error;

/// Tooling-specific errors.
#[derive(Debug, Error)]
pub enum CliError {
    /// Compilation failed (Core).
    #[error("{message}")]
    Compilation {
        message: String,
        #[source]
        source: Option<Box<crate::error::compile::CompileError>>,
    },

    /// Compilation failed (Compiler).
    #[error("{message}")]
    Compiler {
        message: String,
        #[source]
        source: Option<Box<crate::error::compiler::CompilerError>>,
    },

    /// Input file not found.
    #[error("Input file not found: {path}")]
    InputNotFound { path: PathBuf },

    /// Failed to write output file.
    #[error("Failed to write output file '{path}': {message}")]
    OutputWrite { path: PathBuf, message: String },

    /// Configuration error.
    #[error("Configuration error: {message}")]
    Config { message: String },

    /// Runtime error.
    #[error("Runtime error: {message}")]
    Runtime { message: String },

    /// Unknown pass name.
    #[error("Unknown pass '{name}': pass not registered or not available")]
    UnknownPass {
        name: String,
        suggestion: Box<Option<Suggestion>>,
        code: ErrorCode,
    },

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl CliError {
    /// Returns the exit code for this error.
    pub fn exit_code(&self) -> i32 {
        match self {
            CliError::InputNotFound { .. } => 66, // EX_NOINPUT
            CliError::OutputWrite { .. } => 73,   // EX_CANTCREAT
            CliError::Compilation { .. } => 1,    // General error
            CliError::Compiler { .. } => 1,       // General error
            CliError::Config { .. } => 78,        // EX_CONFIG
            CliError::Runtime { .. } => 1,        // General error
            CliError::UnknownPass { .. } => 64,   // EX_USAGE
            CliError::Io(_) => 74,                // EX_IOERR
            CliError::Json(_) => 65,              // EX_DATAERR
        }
    }

    /// Creates a compilation error from a CompileError.
    pub fn from_compile_error(err: crate::error::compile::CompileError) -> Self {
        CliError::Compilation {
            message: err.to_string(),
            source: Some(Box::new(err)),
        }
    }

    /// Creates a compilation error from a CompilerError.
    pub fn from_compiler_error(err: crate::error::compiler::CompilerError) -> Self {
        CliError::Compiler {
            message: err.to_string(),
            source: Some(Box::new(err)),
        }
    }

    /// Creates an unknown pass error with an optional suggestion.
    pub fn unknown_pass(name: impl Into<String>, suggestion: Option<Suggestion>) -> Self {
        CliError::UnknownPass {
            name: name.into(),
            suggestion: Box::new(suggestion),
            code: ErrorCode::PassNotFound,
        }
    }

    /// Get suggestion for this error, if available.
    pub fn suggestion(&self) -> Option<&Suggestion> {
        match self {
            CliError::UnknownPass { suggestion, .. } => suggestion.as_ref().as_ref(),
            _ => None,
        }
    }
}

/// Result type for tooling operations.
pub type CliResult<T> = Result<T, CliError>;
