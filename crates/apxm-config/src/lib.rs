//! Configuration primitives for APxM tooling and runtimes.
//!
//! This crate parses the TOML-based `~/.apxm/config.toml` (and project-specific variants)
//! so that the CLI, runtime, and future tooling can load provider definitions, capability
//! metadata, tool guards, and execpolicy references from a single schema.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use dirs::home_dir;
use serde::{Deserialize, Serialize};
use std::env;
use thiserror::Error;

pub(crate) type Result<T> = std::result::Result<T, ConfigError>;

/// Application configuration loaded from TOML files.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ApXmConfig {
    /// Chat/runtime specific flags.
    pub chat: ChatConfig,

    /// LLM backend definitions.
    pub llm_backends: Vec<LlmBackendConfig>,

    /// Capability definitions (e.g., tool metadata files).
    pub capabilities: Vec<CapabilityConfig>,

    /// Tool-specific behavior overrides.
    pub tools: HashMap<String, ToolConfig>,

    /// Exec policy references.
    pub execpolicy: ExecPolicyConfig,
}

/// Configuration for the chat/runtime surface.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatConfig {
    /// Explicit LLM providers to load.
    pub providers: Vec<String>,

    /// Default exec policy (e.g., `project:execpolicy.toml`).
    pub default_exec_policy: Option<String>,

    /// Optional default model identifier.
    pub default_model: Option<String>,
}

/// Definition of an LLM backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmBackendConfig {
    /// Identifier used in the configuration (e.g., `openai`, `ollama-local`).
    pub name: String,

    /// Human-friendly provider name (optional).
    pub provider: Option<String>,

    /// Model alias to request.
    pub model: Option<String>,

    /// API key or token (safely stored in config).
    pub api_key: Option<String>,

    /// URL/endpoint for the provider service.
    pub endpoint: Option<String>,

    /// Arbitrary backend options.
    #[serde(default)]
    pub options: HashMap<String, String>,
}

impl Default for LlmBackendConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            provider: None,
            model: None,
            api_key: None,
            endpoint: None,
            options: HashMap::new(),
        }
    }
}

/// Capability descriptor used when loading capability metadata.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CapabilityConfig {
    /// Capability name (as registered on the runtime).
    pub name: String,
    /// If provided, the path to a JSON schema that validates inputs.
    pub schema_path: Option<PathBuf>,
    /// Path to the binary/module that implements the capability.
    pub module: Option<PathBuf>,
    /// Enable or disable the capability in this context.
    pub enabled: Option<bool>,
}

/// Tool-specific configuration overrides.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolConfig {
    /// Whether the tool is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Trusted folders for this tool (used by execpolicy).
    #[serde(default)]
    pub trusted_folders: Vec<PathBuf>,

    /// Optional policy name to use when the tool is invoked.
    pub policy: Option<String>,
}

/// Exec policy support (filenames, trusted descriptors, etc.).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExecPolicyConfig {
    /// Explicit execpolicy files to load (e.g., `project:execpolicy.toml`).
    #[serde(default)]
    pub policy_files: Vec<String>,

    /// Optional fallback/default policy identifier.
    pub default_policy: Option<String>,
}

fn default_true() -> bool {
    true
}

impl ApXmConfig {
    /// Loads configuration from the given path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path).map_err(ConfigError::Io)?;
        toml::from_str::<ApXmConfig>(&contents).map_err(ConfigError::Parse)
    }

    /// Returns the default configuration path (`$HOME/.apxm/config.toml`).
    pub fn default_path() -> Result<PathBuf> {
        let home = home_dir().ok_or(ConfigError::HomeDirMissing)?;
        Ok(home.join(".apxm").join("config.toml"))
    }

    /// Load configuration from the default location.
    pub fn load_default() -> Result<Self> {
        let path = Self::default_path()?;
        Self::from_file(path)
    }

    /// Load configuration for the current working directory, falling back to the
    /// global config when no project-level file exists.
    pub fn load_scoped() -> Result<Self> {
        if let Some(path) = project_config_path() {
            return Self::from_file(path);
        }
        Self::load_default()
    }
}

fn project_config_path() -> Option<PathBuf> {
    let cwd = env::current_dir().ok()?;
    for ancestor in cwd.ancestors() {
        let candidate = ancestor.join(".apxm").join("config.toml");
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// Errors that can occur while parsing APxM configuration files.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("IO failure when reading config: {0}")]
    Io(#[from] std::io::Error),

    #[error("Failed to parse TOML config: {0}")]
    Parse(#[from] toml::de::Error),

    #[error("Unable to determine home directory for default config path")]
    HomeDirMissing,
}

#[cfg(test)]
mod tests {
    use super::*;
    use dirs::home_dir;
    use std::env;

    #[test]
    fn deserialize_basic_config() {
        let toml = r#"
            [chat]
            providers = ["openai", "local"]
            default_exec_policy = "project:policy.toml"

            [[llm_backends]]
            name = "openai"
            provider = "openai"
            model = "gpt-4"
            api_key = "token"

            [tools.shell]
            enabled = true
            trusted_folders = ["/home/work"]
        "#;

        let config: ApXmConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.chat.providers.len(), 2);
        assert_eq!(
            config.chat.default_exec_policy.as_deref(),
            Some("project:policy.toml")
        );
        assert_eq!(config.llm_backends.first().unwrap().name, "openai");
        assert!(config.tools.contains_key("shell"));
    }

    #[test]
    fn default_path_respects_home() {
        let home = env::var("HOME").expect("HOME must be set for this test");
        let expected = PathBuf::from(home).join(".apxm").join("config.toml");
        assert_eq!(ApXmConfig::default_path().unwrap(), expected);
    }

    #[test]
    fn error_when_home_missing() {
        let original = env::var("HOME").ok();
        unsafe {
            env::remove_var("HOME");
        }
        if home_dir().is_some() {
            if let Some(value) = original {
                unsafe {
                    env::set_var("HOME", value);
                }
            }
            return;
        }
        let res = ApXmConfig::default_path();
        assert!(matches!(res, Err(ConfigError::HomeDirMissing)));
        if let Some(value) = original {
            unsafe {
                env::set_var("HOME", value);
            }
        }
    }
}
