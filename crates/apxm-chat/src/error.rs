//! Error types for the chat system

use apxm_core::error::RuntimeError;
use apxm_linker::error::LinkerError;

/// Chat error types
#[derive(Debug, thiserror::Error)]
pub enum ChatError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Storage error
    #[error("Storage error: {0}")]
    Storage(#[from] RuntimeError),

    /// Model error
    #[error("Model error: {message}")]
    Model {
        message: String,
        #[source]
        source: Option<anyhow::Error>,
    },

    /// Linker error
    #[error("Linker error: {0}")]
    Linker(#[from] LinkerError),

    /// Translation error
    #[error("Translation error: {0}")]
    Translation(String),

    /// Command error
    #[error("Command error: {0}")]
    Command(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Result type for chat operations
pub type ChatResult<T> = Result<T, ChatError>;

impl ChatError {
    /// Wrap a low-level model failure with a user-facing message.
    pub fn model_failure(message: impl Into<String>, source: anyhow::Error) -> Self {
        ChatError::Model {
            message: message.into(),
            source: Some(source),
        }
    }
}

impl From<anyhow::Error> for ChatError {
    fn from(err: anyhow::Error) -> Self {
        ChatError::Model {
            message: "Model backend request failed".to_string(),
            source: Some(err),
        }
    }
}
