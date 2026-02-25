//! Data-driven provider metadata.
//!
//! [`ProviderSpec`] replaces enum-based provider proliferation with a single,
//! extensible metadata structure. Each provider is described by its protocol,
//! API key requirements, and default URLs.

use serde::{Deserialize, Serialize};

/// The wire protocol a provider speaks.
///
/// This determines which backend implementation handles requests.
/// For example, OpenRouter uses [`ProviderProtocol::OpenAI`] because
/// it's OpenAI-compatible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderProtocol {
    /// OpenAI-compatible API (also used by OpenRouter, Together, etc.)
    OpenAI,
    /// Anthropic Messages API
    Anthropic,
    /// Google Gemini API
    Google,
    /// Ollama local API
    Ollama,
}

impl ProviderProtocol {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProviderProtocol::OpenAI => "openai",
            ProviderProtocol::Anthropic => "anthropic",
            ProviderProtocol::Google => "google",
            ProviderProtocol::Ollama => "ollama",
        }
    }
}

impl std::fmt::Display for ProviderProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for ProviderProtocol {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(ProviderProtocol::OpenAI),
            "anthropic" => Ok(ProviderProtocol::Anthropic),
            "google" => Ok(ProviderProtocol::Google),
            "ollama" => Ok(ProviderProtocol::Ollama),
            _ => Err(format!("Unknown provider protocol: '{}'", s)),
        }
    }
}

/// Data-driven provider metadata.
///
/// Describes a provider's identity, API requirements, and wire protocol
/// without requiring a new enum variant for each provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderSpec {
    /// Canonical provider identifier (e.g., "openai", "anthropic", "openrouter").
    pub id: String,
    /// Environment variable name for the API key (e.g., "OPENAI_API_KEY").
    pub api_key_env_var: Option<String>,
    /// Default base URL (e.g., "http://localhost:11434" for Ollama).
    pub default_base_url: Option<String>,
    /// Whether this provider requires an API key to function.
    pub requires_api_key: bool,
    /// The wire protocol this provider speaks.
    pub protocol: ProviderProtocol,
    /// Alternative names for this provider (e.g., "gemini" → Google).
    #[serde(default)]
    pub aliases: Vec<String>,
}

/// Built-in provider specifications.
pub const BUILTIN_PROVIDERS: &[BuiltinProviderSpec] = &[
    BuiltinProviderSpec {
        id: "ollama",
        api_key_env_var: Some("OLLAMA_API_KEY"),
        default_base_url: Some("http://localhost:11434"),
        requires_api_key: false,
        protocol: ProviderProtocol::Ollama,
        aliases: &[],
    },
    BuiltinProviderSpec {
        id: "openai",
        api_key_env_var: Some("OPENAI_API_KEY"),
        default_base_url: None,
        requires_api_key: true,
        protocol: ProviderProtocol::OpenAI,
        aliases: &[],
    },
    BuiltinProviderSpec {
        id: "anthropic",
        api_key_env_var: Some("ANTHROPIC_API_KEY"),
        default_base_url: None,
        requires_api_key: true,
        protocol: ProviderProtocol::Anthropic,
        aliases: &[],
    },
    BuiltinProviderSpec {
        id: "google",
        api_key_env_var: Some("GOOGLE_API_KEY"),
        default_base_url: None,
        requires_api_key: true,
        protocol: ProviderProtocol::Google,
        aliases: &["gemini"],
    },
    BuiltinProviderSpec {
        id: "openrouter",
        api_key_env_var: Some("OPENROUTER_API_KEY"),
        default_base_url: Some("https://openrouter.ai/api/v1"),
        requires_api_key: true,
        protocol: ProviderProtocol::OpenAI,
        aliases: &[],
    },
];

/// Static provider spec (const-friendly, no heap allocations).
#[derive(Debug, Clone, Copy)]
pub struct BuiltinProviderSpec {
    pub id: &'static str,
    pub api_key_env_var: Option<&'static str>,
    pub default_base_url: Option<&'static str>,
    pub requires_api_key: bool,
    pub protocol: ProviderProtocol,
    pub aliases: &'static [&'static str],
}

impl BuiltinProviderSpec {
    /// Convert to an owned `ProviderSpec`.
    pub fn to_provider_spec(&self) -> ProviderSpec {
        ProviderSpec {
            id: self.id.to_string(),
            api_key_env_var: self.api_key_env_var.map(|s| s.to_string()),
            default_base_url: self.default_base_url.map(|s| s.to_string()),
            requires_api_key: self.requires_api_key,
            protocol: self.protocol,
            aliases: self.aliases.iter().map(|s| s.to_string()).collect(),
        }
    }
}

/// Look up a built-in provider by id or alias.
///
/// Returns `None` if the provider is not built-in.
pub fn resolve_builtin_provider(name: &str) -> Option<&'static BuiltinProviderSpec> {
    let lower = name.to_lowercase();
    BUILTIN_PROVIDERS
        .iter()
        .find(|spec| spec.id == lower || spec.aliases.iter().any(|alias| *alias == lower))
}

/// Resolve a provider spec from name, checking builtins first.
///
/// For custom providers, the caller should check their config after this returns `None`.
pub fn resolve_provider_spec(name: &str) -> Option<ProviderSpec> {
    resolve_builtin_provider(name).map(|b| b.to_provider_spec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_builtin_by_id() {
        let spec = resolve_builtin_provider("openai").unwrap();
        assert_eq!(spec.id, "openai");
        assert_eq!(spec.protocol, ProviderProtocol::OpenAI);
        assert!(spec.requires_api_key);
    }

    #[test]
    fn test_resolve_builtin_by_alias() {
        let spec = resolve_builtin_provider("gemini").unwrap();
        assert_eq!(spec.id, "google");
        assert_eq!(spec.protocol, ProviderProtocol::Google);
    }

    #[test]
    fn test_resolve_builtin_case_insensitive() {
        assert!(resolve_builtin_provider("OpenAI").is_some());
        assert!(resolve_builtin_provider("ANTHROPIC").is_some());
        assert!(resolve_builtin_provider("Gemini").is_some());
    }

    #[test]
    fn test_resolve_unknown_returns_none() {
        assert!(resolve_builtin_provider("unknown_provider").is_none());
    }

    #[test]
    fn test_openrouter_uses_openai_protocol() {
        let spec = resolve_builtin_provider("openrouter").unwrap();
        assert_eq!(spec.protocol, ProviderProtocol::OpenAI);
        assert_eq!(spec.default_base_url, Some("https://openrouter.ai/api/v1"));
    }

    #[test]
    fn test_ollama_no_api_key_required() {
        let spec = resolve_builtin_provider("ollama").unwrap();
        assert!(!spec.requires_api_key);
        assert_eq!(spec.default_base_url, Some("http://localhost:11434"));
    }

    #[test]
    fn test_provider_protocol_roundtrip() {
        let proto = ProviderProtocol::OpenAI;
        assert_eq!(proto.as_str().parse::<ProviderProtocol>().unwrap(), proto);
    }

    #[test]
    fn test_to_provider_spec() {
        let builtin = resolve_builtin_provider("google").unwrap();
        let spec = builtin.to_provider_spec();
        assert_eq!(spec.id, "google");
        assert_eq!(spec.aliases, vec!["gemini"]);
        assert_eq!(spec.protocol, ProviderProtocol::Google);
    }

    #[test]
    fn test_provider_spec_serde() {
        let spec = ProviderSpec {
            id: "together".to_string(),
            api_key_env_var: Some("TOGETHER_API_KEY".to_string()),
            default_base_url: Some("https://api.together.xyz/v1".to_string()),
            requires_api_key: true,
            protocol: ProviderProtocol::OpenAI,
            aliases: vec![],
        };

        let json = serde_json::to_string(&spec).unwrap();
        let deserialized: ProviderSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "together");
        assert_eq!(deserialized.protocol, ProviderProtocol::OpenAI);
    }

    #[test]
    fn test_builtin_providers_all_have_unique_ids() {
        let mut ids: Vec<&str> = BUILTIN_PROVIDERS.iter().map(|s| s.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), BUILTIN_PROVIDERS.len());
    }
}
