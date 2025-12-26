//! LLM backend implementations and traits.
//!
//! This module provides:
//! - `LLMBackend` trait: Unified interface for all providers
//! - Request/Response types: Normalized API across providers
//! - Provider implementations: OpenAI, Anthropic, Google, Ollama
//! - Factory: Create backends from configuration

pub mod request;
pub mod response;
pub mod traits;

pub mod anthropic;
pub mod google;
pub mod ollama;
pub mod openai;

pub use anthropic::{AnthropicBackend, AnthropicModel};
pub use google::{GoogleBackend, GoogleModel};
pub use ollama::{OllamaBackend, OllamaModel};
pub use openai::{OpenAIBackend, OpenAIModel};
pub use request::{GenerationConfig, LLMRequest, RequestBuilder};
pub use response::{LLMResponse, TokenUsage};
pub use traits::LLMBackend;

use std::sync::Arc;

/// Factory for creating LLM backends from provider configuration.
pub struct BackendFactory;

impl BackendFactory {
    /// Create a backend from provider type and API key.
    pub async fn create(
        provider: &str,
        api_key: &str,
        config: Option<serde_json::Value>,
    ) -> anyhow::Result<Arc<dyn LLMBackend>> {
        match provider.to_lowercase().as_str() {
            "openai" => {
                let backend = openai::OpenAIBackend::new(api_key, config).await?;
                Ok(Arc::new(backend))
            }
            "anthropic" => {
                let backend = anthropic::AnthropicBackend::new(api_key, config).await?;
                Ok(Arc::new(backend))
            }
            "google" => {
                let backend = google::GoogleBackend::new(api_key, config).await?;
                Ok(Arc::new(backend))
            }
            "ollama" => {
                let backend = ollama::OllamaBackend::new(api_key, config).await?;
                Ok(Arc::new(backend))
            }
            _ => Err(anyhow::anyhow!(
                "Unknown provider: {}. Supported: openai, anthropic, google, ollama",
                provider
            )),
        }
    }

    /// List available backend providers.
    pub fn list_providers() -> Vec<&'static str> {
        vec!["openai", "anthropic", "google", "ollama"]
    }
}
