//! Error types for the driver module.

use apxm_core::error::{RuntimeError, compiler::CompilerError};
use thiserror::Error;

use crate::config::ConfigError;

/// Errors produced by the driver when coordinating compiler/runtime.
#[derive(Debug, Error)]
pub enum DriverError {
    #[error("Compiler error: {0}")]
    Compiler(#[from] CompilerError),

    #[error("Runtime error: {0}")]
    Runtime(#[from] RuntimeError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Driver error: {0}")]
    Driver(String),
}
