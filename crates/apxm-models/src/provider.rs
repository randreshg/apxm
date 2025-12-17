//! Provider enum for unified backend access.
//!
//! Provides an enum-based dispatch system similar to Zed's architecture.

use crate::backends::{
    AnthropicBackend, GoogleBackend, LLMBackend, LLMRequest, LLMResponse, OllamaBackend,
    OpenAIBackend,
};
use apxm_core::types::{ModelCapabilities, ModelInfo};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Unified provider enum containing all supported backends.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ProviderId {
    OpenAI,
    Anthropic,
    Google,
    Ollama,
}

impl ProviderId {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProviderId::OpenAI => "openai",
            ProviderId::Anthropic => "anthropic",
            ProviderId::Google => "google",
            ProviderId::Ollama => "ollama",
        }
    }
}

impl std::fmt::Display for ProviderId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for ProviderId {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(ProviderId::OpenAI),
            "anthropic" => Ok(ProviderId::Anthropic),
            "google" => Ok(ProviderId::Google),
            "ollama" => Ok(ProviderId::Ollama),
            _ => Err(anyhow::anyhow!("Unknown provider: {}", s)),
        }
    }
}

/// Enum wrapping all backend implementations.
pub enum Provider {
    OpenAI(OpenAIBackend),
    Anthropic(AnthropicBackend),
    Google(GoogleBackend),
    Ollama(OllamaBackend),
}

impl Provider {
    /// Create a provider from ID and configuration.
    pub async fn new(
        provider_id: ProviderId,
        api_key: &str,
        config: Option<serde_json::Value>,
    ) -> anyhow::Result<Self> {
        match provider_id {
            ProviderId::OpenAI => Ok(Provider::OpenAI(OpenAIBackend::new(api_key, config).await?)),
            ProviderId::Anthropic => Ok(Provider::Anthropic(
                AnthropicBackend::new(api_key, config).await?,
            )),
            ProviderId::Google => Ok(Provider::Google(GoogleBackend::new(api_key, config).await?)),
            ProviderId::Ollama => Ok(Provider::Ollama(OllamaBackend::new(api_key, config).await?)),
        }
    }

    /// Get the provider ID.
    pub fn provider_id(&self) -> ProviderId {
        match self {
            Provider::OpenAI(_) => ProviderId::OpenAI,
            Provider::Anthropic(_) => ProviderId::Anthropic,
            Provider::Google(_) => ProviderId::Google,
            Provider::Ollama(_) => ProviderId::Ollama,
        }
    }
}

#[async_trait]
impl LLMBackend for Provider {
    async fn generate(&self, request: LLMRequest) -> anyhow::Result<LLMResponse> {
        match self {
            Provider::OpenAI(backend) => backend.generate(request).await,
            Provider::Anthropic(backend) => backend.generate(request).await,
            Provider::Google(backend) => backend.generate(request).await,
            Provider::Ollama(backend) => backend.generate(request).await,
        }
    }

    fn name(&self) -> &str {
        match self {
            Provider::OpenAI(backend) => backend.name(),
            Provider::Anthropic(backend) => backend.name(),
            Provider::Google(backend) => backend.name(),
            Provider::Ollama(backend) => backend.name(),
        }
    }

    fn model(&self) -> &str {
        match self {
            Provider::OpenAI(backend) => backend.model(),
            Provider::Anthropic(backend) => backend.model(),
            Provider::Google(backend) => backend.model(),
            Provider::Ollama(backend) => backend.model(),
        }
    }

    async fn health_check(&self) -> anyhow::Result<()> {
        match self {
            Provider::OpenAI(backend) => backend.health_check().await,
            Provider::Anthropic(backend) => backend.health_check().await,
            Provider::Google(backend) => backend.health_check().await,
            Provider::Ollama(backend) => backend.health_check().await,
        }
    }

    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        match self {
            Provider::OpenAI(backend) => backend.list_models().await,
            Provider::Anthropic(backend) => backend.list_models().await,
            Provider::Google(backend) => backend.list_models().await,
            Provider::Ollama(backend) => backend.list_models().await,
        }
    }

    fn capabilities(&self) -> ModelCapabilities {
        match self {
            Provider::OpenAI(backend) => backend.capabilities(),
            Provider::Anthropic(backend) => backend.capabilities(),
            Provider::Google(backend) => backend.capabilities(),
            Provider::Ollama(backend) => backend.capabilities(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_id_parsing() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!("openai".parse::<ProviderId>()?, ProviderId::OpenAI);
        assert_eq!("anthropic".parse::<ProviderId>()?, ProviderId::Anthropic);
        assert_eq!("google".parse::<ProviderId>()?, ProviderId::Google);
        assert_eq!("ollama".parse::<ProviderId>()?, ProviderId::Ollama);

        assert!("unknown".parse::<ProviderId>().is_err());
        Ok(())
    }

    #[test]
    fn test_provider_id_to_string() {
        assert_eq!(ProviderId::OpenAI.as_str(), "openai");
        assert_eq!(ProviderId::Anthropic.as_str(), "anthropic");
        assert_eq!(ProviderId::Google.as_str(), "google");
        assert_eq!(ProviderId::Ollama.as_str(), "ollama");
    }
}
