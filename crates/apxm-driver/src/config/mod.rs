//! Configuration primitives for APxM tooling and runtimes.
//!
//! This module parses the TOML-based `~/.apxm/config.toml` (and project-specific variants)
//! so that the driver, runtime, and future tooling can load provider definitions, capability
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

    /// System prompts for LLM operations (ask, think, reason, plan, reflect).
    #[serde(default)]
    pub instruction: InstructionConfig,
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

    /// Path to session storage directory.
    #[serde(default)]
    pub session_storage: Option<PathBuf>,

    /// Maximum context tokens for chat sessions.
    #[serde(default = "default_max_context_tokens")]
    pub max_context_tokens: usize,

    /// Model to use for planning (defaults to default_model if not specified).
    pub planning_model: Option<String>,

    /// System prompt for chat sessions.
    pub system_prompt: Option<String>,
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

    /// Commands to block (bash).
    #[serde(default)]
    pub blocked_commands: Vec<String>,

    /// Commands to allow (bash whitelist mode).
    #[serde(default)]
    pub allowed_commands: Vec<String>,

    /// Paths to block (read/write).
    #[serde(default)]
    pub blocked_paths: Vec<PathBuf>,

    /// Paths to allow (read/write).
    #[serde(default)]
    pub allowed_paths: Vec<PathBuf>,

    /// File extensions to allow (read/write).
    #[serde(default)]
    pub allowed_extensions: Vec<String>,

    /// File extensions to block (write).
    #[serde(default)]
    pub blocked_extensions: Vec<String>,

    /// Query terms to block (search_web).
    #[serde(default)]
    pub blocked_queries: Vec<String>,

    /// Domains to allow (search_web).
    #[serde(default)]
    pub allowed_domains: Vec<String>,

    /// Domains to block (search_web).
    #[serde(default)]
    pub blocked_domains: Vec<String>,

    /// Working directory (bash) or base directory (read/write).
    pub working_directory: Option<PathBuf>,

    /// Timeout in seconds (bash).
    pub timeout_secs: Option<u64>,

    /// Maximum output bytes (bash).
    pub max_output_bytes: Option<usize>,

    /// Maximum file size bytes (read/write).
    pub max_file_size: Option<usize>,

    /// Maximum results (search_web).
    pub max_results: Option<usize>,

    /// Safe search toggle (search_web).
    pub safe_search: Option<bool>,

    /// Search depth mode (search_web).
    pub search_depth: Option<apxm_tools::SearchDepth>,

    /// Endpoint override (search_web).
    pub endpoint: Option<String>,

    /// Include answer summary (search_web).
    pub include_answer: Option<bool>,

    /// Create parent directories automatically (write).
    pub create_directories: Option<bool>,

    /// Overwrite existing files (write).
    pub overwrite_existing: Option<bool>,

    /// Maximum default lines for reads.
    pub max_default_lines: Option<usize>,
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

// Re-export InstructionConfig from apxm-core for consistency
pub use apxm_core::InstructionConfig;

fn default_true() -> bool {
    true
}

fn default_max_context_tokens() -> usize {
    8192
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

    /// Build APxM standard tool configuration from config file fields.
    pub fn tools_config(&self) -> apxm_tools::ToolsConfig {
        let mut tools_config = apxm_tools::ToolsConfig::default();

        for capability in &self.capabilities {
            if let Some(enabled) = capability.enabled {
                apply_enabled_override(&capability.name, enabled, &mut tools_config);
            }
        }

        for (tool_name, tool_config) in &self.tools {
            apply_tool_preset(tool_name, &mut tools_config);
            apply_enabled_override(tool_name, tool_config.enabled, &mut tools_config);
            apply_tool_overrides(tool_name, tool_config, &mut tools_config);
        }

        tools_config
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

fn apply_enabled_override(name: &str, enabled: bool, config: &mut apxm_tools::ToolsConfig) {
    match normalize_tool_name(name) {
        Some("bash") => config.bash.enabled = enabled,
        Some("read") => config.read.enabled = enabled,
        Some("write") => config.write.enabled = enabled,
        Some("search_web") => config.search_web.enabled = enabled,
        _ => {}
    }
}

fn apply_tool_preset(name: &str, config: &mut apxm_tools::ToolsConfig) {
    match name {
        "bash_safe" => {
            config.bash.blocked_commands = vec![
                "rm -rf".to_string(),
                "rm -r".to_string(),
                "sudo".to_string(),
                "su ".to_string(),
                "mkfs".to_string(),
                "fdisk".to_string(),
                "dd if=".to_string(),
            ];
            config.bash.allowed_commands = None;
            config.bash.timeout_secs = 120;
            config.bash.max_output_bytes = 100_000;
            config.bash.enabled = true;
        }
        "bash_build" => {
            config.bash.blocked_commands =
                vec!["sudo".to_string(), "su ".to_string(), "rm -rf".to_string()];
            config.bash.allowed_commands = None;
            config.bash.timeout_secs = 600;
            config.bash.enabled = true;
        }
        "bash_git" => {
            config.bash.allowed_commands = Some(vec![
                "git status".to_string(),
                "git diff".to_string(),
                "git log".to_string(),
                "git show".to_string(),
                "git add".to_string(),
                "git reset".to_string(),
                "git commit".to_string(),
                "git push".to_string(),
                "git pull".to_string(),
                "git branch".to_string(),
                "git checkout".to_string(),
                "git switch".to_string(),
                "git merge".to_string(),
                "git rebase".to_string(),
                "git stash".to_string(),
                "git fetch".to_string(),
                "git remote".to_string(),
                "git clone".to_string(),
                "git rev-parse".to_string(),
                "git config --get".to_string(),
                "git config --list".to_string(),
            ]);
            config.bash.enabled = true;
        }
        "read_source" => {
            config.read.allowed_extensions = Some(
                vec![
                    "rs", "py", "js", "ts", "tsx", "jsx", "go", "java", "c", "cpp", "h", "hpp",
                    "toml", "yaml", "yml", "json", "md", "txt", "sh", "css", "scss", "html", "sql",
                    "graphql", "proto", "xml", "lock", "mod", "sum",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
            );
            config.read.blocked_paths = vec![
                PathBuf::from(".env"),
                PathBuf::from(".secret"),
                PathBuf::from("credentials"),
                PathBuf::from(".git/config"),
            ];
            config.read.enabled = true;
        }
        "write_safe" => {
            config.write.blocked_extensions = vec![
                "exe", "com", "msi", "app", "dmg", "sh", "bat", "ps1", "cmd", "vbs", "dll", "so",
                "dylib", "bin",
            ]
            .into_iter()
            .map(str::to_string)
            .collect();
            config.write.enabled = true;
        }
        "search_docs" => {
            config.search_web.allowed_domains = Some(
                vec![
                    "docs.rs",
                    "doc.rust-lang.org",
                    "crates.io",
                    "docs.python.org",
                    "pypi.org",
                    "developer.mozilla.org",
                    "nodejs.org",
                    "pkg.go.dev",
                    "learn.microsoft.com",
                    "docs.github.com",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
            );
            config.search_web.max_results = 10;
            config.search_web.safe_search = true;
            config.search_web.search_depth = apxm_tools::SearchDepth::Basic;
            config.search_web.enabled = true;
        }
        "search_research" => {
            config.search_web.max_results = 15;
            config.search_web.safe_search = true;
            config.search_web.search_depth = apxm_tools::SearchDepth::Advanced;
            config.search_web.enabled = true;
        }
        _ => {}
    }
}

fn apply_tool_overrides(
    tool_name: &str,
    tool_config: &ToolConfig,
    config: &mut apxm_tools::ToolsConfig,
) {
    match normalize_tool_name(tool_name) {
        Some("bash") => {
            if !tool_config.blocked_commands.is_empty() {
                config.bash.blocked_commands = tool_config.blocked_commands.clone();
            }
            if !tool_config.allowed_commands.is_empty() {
                config.bash.allowed_commands = Some(tool_config.allowed_commands.clone());
            }
            if let Some(working_directory) = &tool_config.working_directory {
                config.bash.working_directory = Some(working_directory.clone());
            } else if !tool_config.trusted_folders.is_empty()
                && config.bash.working_directory.is_none()
            {
                config.bash.working_directory = tool_config.trusted_folders.first().cloned();
            }
            if let Some(timeout_secs) = tool_config.timeout_secs {
                config.bash.timeout_secs = timeout_secs;
            }
            if let Some(max_output_bytes) = tool_config.max_output_bytes {
                config.bash.max_output_bytes = max_output_bytes;
            }
        }
        Some("read") => {
            if !tool_config.blocked_paths.is_empty() {
                config.read.blocked_paths = tool_config.blocked_paths.clone();
            }
            if !tool_config.allowed_paths.is_empty() {
                config.read.allowed_paths = Some(tool_config.allowed_paths.clone());
            } else if !tool_config.trusted_folders.is_empty() {
                config.read.allowed_paths = Some(tool_config.trusted_folders.clone());
            }
            if !tool_config.allowed_extensions.is_empty() {
                config.read.allowed_extensions = Some(tool_config.allowed_extensions.clone());
            }
            if let Some(max_file_size) = tool_config.max_file_size {
                config.read.max_file_size = max_file_size;
            }
            if let Some(base_directory) = &tool_config.working_directory {
                config.read.base_directory = Some(base_directory.clone());
            }
            if let Some(max_default_lines) = tool_config.max_default_lines {
                config.read.max_default_lines = max_default_lines;
            }
        }
        Some("write") => {
            if !tool_config.blocked_paths.is_empty() {
                config.write.blocked_paths = tool_config.blocked_paths.clone();
            }
            if !tool_config.allowed_paths.is_empty() {
                config.write.allowed_paths = Some(tool_config.allowed_paths.clone());
            } else if !tool_config.trusted_folders.is_empty() {
                config.write.allowed_paths = Some(tool_config.trusted_folders.clone());
            }
            if !tool_config.allowed_extensions.is_empty() {
                config.write.allowed_extensions = Some(tool_config.allowed_extensions.clone());
            }
            if !tool_config.blocked_extensions.is_empty() {
                config.write.blocked_extensions = tool_config.blocked_extensions.clone();
            }
            if let Some(max_file_size) = tool_config.max_file_size {
                config.write.max_file_size = Some(max_file_size);
            }
            if let Some(base_directory) = &tool_config.working_directory {
                config.write.base_directory = Some(base_directory.clone());
            }
            if let Some(create_directories) = tool_config.create_directories {
                config.write.create_directories = create_directories;
            }
            if let Some(overwrite_existing) = tool_config.overwrite_existing {
                config.write.overwrite_existing = overwrite_existing;
            }
        }
        Some("search_web") => {
            if !tool_config.allowed_domains.is_empty() {
                config.search_web.allowed_domains = Some(tool_config.allowed_domains.clone());
            }
            if !tool_config.blocked_domains.is_empty() {
                config.search_web.blocked_domains = tool_config.blocked_domains.clone();
            }
            if !tool_config.blocked_queries.is_empty() {
                config.search_web.blocked_queries = tool_config.blocked_queries.clone();
            }
            if let Some(max_results) = tool_config.max_results {
                config.search_web.max_results = max_results;
            }
            if let Some(safe_search) = tool_config.safe_search {
                config.search_web.safe_search = safe_search;
            }
            if let Some(search_depth) = tool_config.search_depth {
                config.search_web.search_depth = search_depth;
            }
            if let Some(endpoint) = &tool_config.endpoint {
                config.search_web.endpoint = endpoint.clone();
            }
            if let Some(include_answer) = tool_config.include_answer {
                config.search_web.include_answer = include_answer;
            }
        }
        _ => {}
    }
}

fn normalize_tool_name(name: &str) -> Option<&'static str> {
    match name {
        "bash" | "shell" | "terminal" | "bash_safe" | "bash_build" | "bash_git" => Some("bash"),
        "read" | "read_file" | "read_source" => Some("read"),
        "write" | "write_file" | "write_safe" => Some("write"),
        "search_web" | "web_search" | "search" | "search_docs" | "search_research" => {
            Some("search_web")
        }
        _ => None,
    }
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
    fn tools_config_applies_presets_and_overrides() {
        let toml = r#"
            [tools.bash_safe]
            enabled = true
            timeout_secs = 42

            [tools.read_source]
            enabled = true
            allowed_paths = ["/repo"]
            max_default_lines = 120

            [tools.search_docs]
            enabled = true
            blocked_queries = ["secrets"]
            max_results = 7
        "#;

        let config: ApXmConfig = toml::from_str(toml).unwrap();
        let tools = config.tools_config();

        assert!(tools.bash.enabled);
        assert_eq!(tools.bash.timeout_secs, 42);
        assert!(
            tools
                .bash
                .blocked_commands
                .iter()
                .any(|command| command == "sudo")
        );

        assert!(tools.read.enabled);
        assert_eq!(
            tools
                .read
                .allowed_paths
                .unwrap_or_default()
                .first()
                .cloned(),
            Some(PathBuf::from("/repo"))
        );
        assert_eq!(tools.read.max_default_lines, 120);
        assert!(
            tools
                .read
                .allowed_extensions
                .unwrap_or_default()
                .iter()
                .any(|ext| ext == "rs")
        );

        assert!(tools.search_web.enabled);
        assert_eq!(tools.search_web.max_results, 7);
        assert!(
            tools
                .search_web
                .blocked_queries
                .iter()
                .any(|term| term == "secrets")
        );
        assert!(tools.search_web.safe_search);
    }

    #[test]
    fn deserialize_instruction_config() {
        let toml = r#"
            [instruction]
            ask = "You are a helpful AI assistant."
            think = "Think step by step."
            reason = "Provide structured reasoning."
            plan = "Create actionable plans."
            reflect = "Analyze execution patterns."
        "#;

        let config: ApXmConfig = toml::from_str(toml).unwrap();
        assert_eq!(
            config.instruction.ask.as_deref(),
            Some("You are a helpful AI assistant.")
        );
        assert_eq!(
            config.instruction.think.as_deref(),
            Some("Think step by step.")
        );
        assert_eq!(
            config.instruction.reason.as_deref(),
            Some("Provide structured reasoning.")
        );
        assert_eq!(
            config.instruction.plan.as_deref(),
            Some("Create actionable plans.")
        );
        assert_eq!(
            config.instruction.reflect.as_deref(),
            Some("Analyze execution patterns.")
        );
    }

    #[test]
    fn instruction_config_defaults_to_none() {
        let toml = r#"
            [chat]
            providers = ["openai"]
        "#;

        let config: ApXmConfig = toml::from_str(toml).unwrap();
        assert!(config.instruction.ask.is_none());
        assert!(config.instruction.think.is_none());
        assert!(config.instruction.reason.is_none());
        assert!(config.instruction.plan.is_none());
        assert!(config.instruction.reflect.is_none());
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
