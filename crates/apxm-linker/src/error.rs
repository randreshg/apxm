use apxm_core::error::{RuntimeError, compiler::CompilerError};
use thiserror::Error;

/// Errors produced by the linker when coordinating compiler/runtime.
#[derive(Debug, Error)]
pub enum LinkerError {
    #[error("Compiler error: {0}")]
    Compiler(#[from] CompilerError),

    #[error("Runtime error: {0}")]
    Runtime(#[from] RuntimeError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Configuration error: {0}")]
    Config(String),
}
