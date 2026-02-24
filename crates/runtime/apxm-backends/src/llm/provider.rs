//! Provider enum for unified backend access.
//!
//! Provides both an enum-based dispatch system and a data-driven
//! [`RegisteredProvider`] for extensible provider management.

use crate::llm::backends::{
    AnthropicBackend, GoogleBackend, LLMBackend, LLMRequest, LLMResponse, OllamaBackend,
    OpenAIBackend,
};
use apxm_core::types::{ModelCapabilities, ModelInfo, ProviderProtocol, ProviderSpec};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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

    /// Convert to a `ProviderProtocol`.
    pub fn to_protocol(&self) -> ProviderProtocol {
        match self {
            ProviderId::OpenAI => ProviderProtocol::OpenAI,
            ProviderId::Anthropic => ProviderProtocol::Anthropic,
            ProviderId::Google => ProviderProtocol::Google,
            ProviderId::Ollama => ProviderProtocol::Ollama,
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

    /// Create a provider from a `ProviderProtocol`.
    ///
    /// This is the data-driven alternative to `new()` — routes by protocol instead
    /// of enum variant, supporting custom/OpenAI-compatible providers.
    pub async fn from_protocol(
        protocol: ProviderProtocol,
        api_key: &str,
        config: Option<serde_json::Value>,
    ) -> anyhow::Result<Self> {
        match protocol {
            ProviderProtocol::OpenAI => {
                Ok(Provider::OpenAI(OpenAIBackend::new(api_key, config).await?))
            }
            ProviderProtocol::Anthropic => Ok(Provider::Anthropic(
                AnthropicBackend::new(api_key, config).await?,
            )),
            ProviderProtocol::Google => {
                Ok(Provider::Google(GoogleBackend::new(api_key, config).await?))
            }
            ProviderProtocol::Ollama => {
                Ok(Provider::Ollama(OllamaBackend::new(api_key, config).await?))
            }
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

/// A provider instance paired with its metadata.
///
/// This is the data-driven alternative to the `Provider` enum, allowing
/// registration of custom/third-party providers without adding enum variants.
pub struct RegisteredProvider {
    /// Provider metadata (protocol, API key env var, etc.).
    pub spec: ProviderSpec,
    /// The actual backend implementation.
    pub backend: Arc<dyn LLMBackend>,
}

impl RegisteredProvider {
    /// Create a registered provider from a spec and backend.
    pub fn new(spec: ProviderSpec, backend: Arc<dyn LLMBackend>) -> Self {
        Self { spec, backend }
    }

    /// Create a registered provider by resolving a `ProviderSpec` and building the backend.
    pub async fn from_spec(
        spec: ProviderSpec,
        api_key: &str,
        config: Option<serde_json::Value>,
    ) -> anyhow::Result<Self> {
        let provider = Provider::from_protocol(spec.protocol, api_key, config).await?;
        let backend: Arc<dyn LLMBackend> = Arc::new(provider);
        Ok(Self { spec, backend })
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

    #[test]
    fn test_provider_id_to_protocol() {
        assert_eq!(ProviderId::OpenAI.to_protocol(), ProviderProtocol::OpenAI);
        assert_eq!(
            ProviderId::Anthropic.to_protocol(),
            ProviderProtocol::Anthropic
        );
        assert_eq!(ProviderId::Google.to_protocol(), ProviderProtocol::Google);
        assert_eq!(ProviderId::Ollama.to_protocol(), ProviderProtocol::Ollama);
    }
}
