//! Configuration for the chat system

use crate::error::{ChatError, ChatResult};
use apxm_config::ApXmConfig;
use std::path::PathBuf;

/// Chat configuration
#[derive(Debug, Clone)]
pub struct ChatConfig {
    /// Base APxM configuration
    pub apxm_config: ApXmConfig,

    /// Path to session storage directory
    pub session_storage_path: PathBuf,

    /// Default model for chat
    pub default_model: String,

    /// Model for planning (defaults to default_model if None)
    pub planning_model: Option<String>,

    /// Maximum context tokens
    pub max_context_tokens: usize,

    /// System prompt
    pub system_prompt: String,
}

impl ChatConfig {
    /// Create ChatConfig from ApXmConfig
    pub fn from_apxm_config(apxm_config: ApXmConfig) -> ChatResult<Self> {
        // Get session storage path from config or use default
        let session_storage_path = if let Some(storage) = &apxm_config.chat.session_storage {
            storage.clone()
        } else {
            dirs::home_dir()
                .ok_or_else(|| ChatError::Config("Could not determine home directory".to_string()))?
                .join(".apxm")
                .join("sessions")
        };

        // Get default model from config or first backend
        let default_model = apxm_config
            .chat
            .default_model
            .clone()
            .or_else(|| {
                apxm_config
                    .llm_backends
                    .first()
                    .map(|backend| backend.name.clone())
            })
            .ok_or_else(|| {
                ChatError::Config(
                    "No LLM backend configured. Add at least one `[[llm_backends]]` entry in `.apxm/config.toml` (see README.md Configuration section).".to_string()
                )
            })?;

        // Get planning model from config (defaults to default_model)
        let planning_model = apxm_config.chat.planning_model.clone();

        // Get max context tokens
        let max_context_tokens = apxm_config.chat.max_context_tokens;

        // Get system prompt
        let system_prompt =
            apxm_config.chat.system_prompt.clone().unwrap_or_else(|| {
                "You are a helpful AI assistant for APxM development.".to_string()
            });

        Ok(Self {
            apxm_config,
            session_storage_path,
            default_model,
            planning_model,
            max_context_tokens,
            system_prompt,
        })
    }

    /// Load scoped configuration (project or global)
    pub fn load_scoped() -> ChatResult<Self> {
        let apxm_config = ApXmConfig::load_scoped().unwrap_or_default();
        Self::from_apxm_config(apxm_config)
    }
}
