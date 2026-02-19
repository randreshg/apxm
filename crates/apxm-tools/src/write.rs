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
use tokio::io::AsyncWriteExt;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WriteConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub blocked_paths: Vec<PathBuf>,
    #[serde(default)]
    pub allowed_paths: Option<Vec<PathBuf>>,
    #[serde(default)]
    pub allowed_extensions: Option<Vec<String>>,
    #[serde(default)]
    pub blocked_extensions: Vec<String>,
    #[serde(default = "default_true")]
    pub create_directories: bool,
    #[serde(default = "default_true")]
    pub overwrite_existing: bool,
    #[serde(default)]
    pub max_file_size: Option<usize>,
    #[serde(default)]
    pub base_directory: Option<PathBuf>,
}

fn default_true() -> bool {
    true
}

impl Default for WriteConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            blocked_paths: Vec::new(),
            allowed_paths: None,
            allowed_extensions: None,
            blocked_extensions: Vec::new(),
            create_directories: true,
            overwrite_existing: true,
            max_file_size: None,
            base_directory: None,
        }
    }
}

pub struct WriteCapability {
    metadata: CapabilityMetadata,
    config: WriteConfig,
}

impl WriteCapability {
    pub fn new() -> Self {
        Self::with_config(WriteConfig::default())
    }

    pub fn with_config(config: WriteConfig) -> Self {
        Self {
            metadata: CapabilityMetadata::new(
                "write",
                "Write content to a file with policy enforcement",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "file_path": { "type": "string" },
                        "content": { "type": "string" },
                        "append": { "type": "boolean", "default": false }
                    },
                    "required": ["file_path", "content"]
                }),
            )
            .with_returns("string")
            .with_latency(30),
            config,
        }
    }

    pub fn safe() -> Self {
        Self::with_config(WriteConfig {
            blocked_extensions: vec![
                "exe", "com", "msi", "app", "dmg", "sh", "bat", "ps1", "cmd", "vbs", "dll", "so",
                "dylib", "bin",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            ..Default::default()
        })
    }

    fn resolve_path(&self, raw_path: &str) -> PathBuf {
        if Path::new(raw_path).is_absolute() {
            return PathBuf::from(raw_path);
        }
        if let Some(base_directory) = &self.config.base_directory {
            return base_directory.join(raw_path);
        }
        PathBuf::from(raw_path)
    }

    fn validate_path_and_content(&self, path: &Path, content_len: usize) -> CapabilityResult<()> {
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

        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or_default();

        if self
            .config
            .blocked_extensions
            .iter()
            .any(|blocked| blocked == extension)
        {
            return Err(RuntimeError::Capability {
                capability: self.metadata.name.clone(),
                message: format!("Extension '{}' is blocked", extension),
            });
        }

        if let Some(allowed_extensions) = &self.config.allowed_extensions
            && !allowed_extensions
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

        if let Some(max_file_size) = self.config.max_file_size
            && content_len > max_file_size
        {
            return Err(RuntimeError::Capability {
                capability: self.metadata.name.clone(),
                message: format!(
                    "Content too large ({} bytes > {} bytes)",
                    content_len, max_file_size
                ),
            });
        }

        Ok(())
    }
}

impl Default for WriteCapability {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CapabilityExecutor for WriteCapability {
    async fn execute(&self, args: HashMap<String, Value>) -> CapabilityResult<Value> {
        let file_path = args
            .get("file_path")
            .or_else(|| args.get("path"))
            .or_else(|| args.get("arg0"))
            .and_then(|value| value.as_string())
            .ok_or_else(|| RuntimeError::Capability {
                capability: self.metadata.name.clone(),
                message: "Missing required 'file_path' argument".to_string(),
            })?;
        let content = args
            .get("content")
            .or_else(|| args.get("arg_content"))
            .or_else(|| args.get("arg1"))
            .and_then(|value| value.as_string())
            .ok_or_else(|| RuntimeError::Capability {
                capability: self.metadata.name.clone(),
                message: "Missing required 'content' argument".to_string(),
            })?;
        let append = args
            .get("append")
            .and_then(|value| value.as_boolean())
            .unwrap_or(false);

        let path = self.resolve_path(file_path);
        self.validate_path_and_content(&path, content.len())?;

        if !self.config.overwrite_existing && !append && path.exists() {
            return Err(RuntimeError::Capability {
                capability: self.metadata.name.clone(),
                message: "File already exists and overwrite_existing=false".to_string(),
            });
        }

        if self.config.create_directories
            && let Some(parent) = path.parent()
        {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|error| RuntimeError::Capability {
                    capability: self.metadata.name.clone(),
                    message: format!(
                        "Failed to create parent directory '{}': {error}",
                        parent.display()
                    ),
                })?;
        }

        if append {
            let mut file = tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .await
                .map_err(|error| RuntimeError::Capability {
                    capability: self.metadata.name.clone(),
                    message: format!("Failed to open file '{}': {error}", path.display()),
                })?;
            file.write_all(content.as_bytes())
                .await
                .map_err(|error| RuntimeError::Capability {
                    capability: self.metadata.name.clone(),
                    message: format!("Failed to append file '{}': {error}", path.display()),
                })?;
        } else {
            tokio::fs::write(&path, content)
                .await
                .map_err(|error| RuntimeError::Capability {
                    capability: self.metadata.name.clone(),
                    message: format!("Failed to write file '{}': {error}", path.display()),
                })?;
        }

        Ok(Value::String(format!(
            "Wrote {} bytes to {}",
            content.len(),
            path.display()
        )))
    }

    fn metadata(&self) -> &CapabilityMetadata {
        &self.metadata
    }
}
