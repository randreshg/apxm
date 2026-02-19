use apxm_core::{error::RuntimeError, types::Value};
use apxm_runtime::capability::{
    executor::{CapabilityExecutor, CapabilityResult},
    metadata::CapabilityMetadata,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReadConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub blocked_paths: Vec<PathBuf>,
    #[serde(default)]
    pub allowed_paths: Option<Vec<PathBuf>>,
    #[serde(default)]
    pub allowed_extensions: Option<Vec<String>>,
    #[serde(default = "default_max_file_size")]
    pub max_file_size: usize,
    #[serde(default)]
    pub base_directory: Option<PathBuf>,
    #[serde(default = "default_max_lines")]
    pub max_default_lines: usize,
}

fn default_true() -> bool {
    true
}

fn default_max_file_size() -> usize {
    1024 * 1024
}

fn default_max_lines() -> usize {
    2000
}

impl Default for ReadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            blocked_paths: Vec::new(),
            allowed_paths: None,
            allowed_extensions: None,
            max_file_size: default_max_file_size(),
            base_directory: None,
            max_default_lines: default_max_lines(),
        }
    }
}

pub struct ReadCapability {
    metadata: CapabilityMetadata,
    config: ReadConfig,
}

impl ReadCapability {
    pub fn new() -> Self {
        Self::with_config(ReadConfig::default())
    }

    pub fn with_config(config: ReadConfig) -> Self {
        Self {
            metadata: CapabilityMetadata::new(
                "read",
                "Read file contents with path and extension restrictions",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "file_path": { "type": "string" },
                        "offset": { "type": "integer", "minimum": 0 },
                        "limit": { "type": "integer", "minimum": 1 }
                    },
                    "required": ["file_path"]
                }),
            )
            .with_returns("string")
            .with_latency(35),
            config,
        }
    }

    pub fn source() -> Self {
        Self::with_config(ReadConfig {
            allowed_extensions: Some(
                vec![
                    "rs", "py", "js", "ts", "tsx", "jsx", "go", "java", "c", "cpp", "h", "hpp",
                    "toml", "yaml", "yml", "json", "md", "txt", "sh", "css", "scss", "html", "sql",
                    "graphql", "proto", "xml", "lock", "mod", "sum",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
            ),
            blocked_paths: vec![
                PathBuf::from(".env"),
                PathBuf::from(".secret"),
                PathBuf::from("credentials"),
                PathBuf::from(".git/config"),
            ],
            ..Default::default()
        })
    }

    fn resolve_path(&self, raw_path: &str) -> PathBuf {
        let raw_path = raw_path.trim();
        if Path::new(raw_path).is_absolute() {
            return PathBuf::from(raw_path);
        }

        if raw_path.starts_with("Users/")
            || raw_path.starts_with("home/")
            || raw_path.starts_with("var/")
            || raw_path.starts_with("tmp/")
            || raw_path.starts_with("etc/")
            || raw_path.starts_with("opt/")
        {
            return PathBuf::from(format!("/{raw_path}"));
        }

        if let Some(base_directory) = &self.config.base_directory {
            return base_directory.join(raw_path);
        }

        PathBuf::from(raw_path)
    }

    fn validate_path(&self, path: &Path) -> CapabilityResult<()> {
        if let Some(blocked_path) = self
            .config
            .blocked_paths
            .iter()
            .find(|blocked| path.starts_with(blocked.as_path()))
        {
            return Err(RuntimeError::Capability {
                capability: self.metadata.name.clone(),
                message: format!("Path blocked by policy: {}", blocked_path.display()),
            });
        }

        if let Some(allowed_paths) = &self.config.allowed_paths
            && !allowed_paths
                .iter()
                .any(|allowed| path.starts_with(allowed.as_path()))
        {
            return Err(RuntimeError::Capability {
                capability: self.metadata.name.clone(),
                message: format!("Path not in allowed list: {}", path.display()),
            });
        }

        if let Some(allowed_extensions) = &self.config.allowed_extensions {
            let extension = path
                .extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or_default();
            if !allowed_extensions
                .iter()
                .any(|allowed| allowed == extension)
            {
                return Err(RuntimeError::Capability {
                    capability: self.metadata.name.clone(),
                    message: format!(
                        "Extension '{}' is not allowed. Allowed: {}",
                        extension,
                        allowed_extensions.join(", ")
                    ),
                });
            }
        }

        Ok(())
    }
}

impl Default for ReadCapability {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CapabilityExecutor for ReadCapability {
    async fn execute(&self, args: HashMap<String, Value>) -> CapabilityResult<Value> {
        let raw_path = args
            .get("file_path")
            .or_else(|| args.get("path"))
            .or_else(|| args.get("arg0"))
            .and_then(|value| value.as_string())
            .ok_or_else(|| RuntimeError::Capability {
                capability: self.metadata.name.clone(),
                message: "Missing required 'file_path' argument".to_string(),
            })?;

        let path = self.resolve_path(raw_path);
        self.validate_path(&path)?;

        let metadata =
            tokio::fs::metadata(&path)
                .await
                .map_err(|error| RuntimeError::Capability {
                    capability: self.metadata.name.clone(),
                    message: format!("Unable to access file '{}': {error}", path.display()),
                })?;

        if metadata.len() as usize > self.config.max_file_size {
            return Err(RuntimeError::Capability {
                capability: self.metadata.name.clone(),
                message: format!(
                    "File exceeds configured limit ({} bytes > {} bytes)",
                    metadata.len(),
                    self.config.max_file_size
                ),
            });
        }

        let content =
            tokio::fs::read_to_string(&path)
                .await
                .map_err(|error| RuntimeError::Capability {
                    capability: self.metadata.name.clone(),
                    message: format!("Unable to read file '{}': {error}", path.display()),
                })?;

        let lines = content.lines().collect::<Vec<_>>();
        let offset = args
            .get("offset")
            .and_then(|value| value.as_u64())
            .unwrap_or(0) as usize;
        let limit = args
            .get("limit")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize)
            .unwrap_or(self.config.max_default_lines);

        if offset >= lines.len() {
            return Ok(Value::String(String::new()));
        }

        let end = (offset + limit).min(lines.len());
        let numbered = lines[offset..end]
            .iter()
            .enumerate()
            .map(|(line_offset, line)| format!("{:>6}\t{}", offset + line_offset + 1, line))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(Value::String(numbered))
    }

    fn metadata(&self) -> &CapabilityMetadata {
        &self.metadata
    }
}
