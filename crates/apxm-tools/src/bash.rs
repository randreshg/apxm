use apxm_core::{error::RuntimeError, types::Value};
use apxm_runtime::capability::{
    executor::{CapabilityExecutor, CapabilityResult},
    metadata::CapabilityMetadata,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::PathBuf, process::Stdio};
use tokio::{process::Command, time::Duration};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BashConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub blocked_commands: Vec<String>,
    #[serde(default)]
    pub allowed_commands: Option<Vec<String>>,
    #[serde(default)]
    pub working_directory: Option<PathBuf>,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    #[serde(default = "default_max_output")]
    pub max_output_bytes: usize,
}

fn default_true() -> bool {
    true
}

fn default_timeout() -> u64 {
    120
}

fn default_max_output() -> usize {
    100_000
}

impl Default for BashConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            blocked_commands: Vec::new(),
            allowed_commands: None,
            working_directory: None,
            timeout_secs: default_timeout(),
            max_output_bytes: default_max_output(),
        }
    }
}

pub struct BashCapability {
    metadata: CapabilityMetadata,
    config: BashConfig,
}

impl BashCapability {
    pub fn new() -> Self {
        Self::with_config(BashConfig::default())
    }

    pub fn with_config(config: BashConfig) -> Self {
        Self {
            metadata: CapabilityMetadata::new(
                "bash",
                "Execute shell commands with policy enforcement",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to run"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Optional timeout in seconds"
                        }
                    },
                    "required": ["command"]
                }),
            )
            .with_returns("string")
            .with_latency(250),
            config,
        }
    }

    pub fn safe() -> Self {
        Self::with_config(BashConfig {
            blocked_commands: vec![
                "rm -rf".to_string(),
                "rm -r".to_string(),
                "sudo".to_string(),
                "su ".to_string(),
                "mkfs".to_string(),
                "fdisk".to_string(),
                "dd if=".to_string(),
            ],
            ..Default::default()
        })
    }

    pub fn build() -> Self {
        Self::with_config(BashConfig {
            blocked_commands: vec!["sudo".to_string(), "su ".to_string(), "rm -rf".to_string()],
            timeout_secs: 600,
            ..Default::default()
        })
    }

    pub fn git() -> Self {
        Self::with_config(BashConfig {
            allowed_commands: Some(vec![
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
            ]),
            ..Default::default()
        })
    }

    fn validate_command(&self, command: &str) -> CapabilityResult<()> {
        if let Some(allowed_commands) = &self.config.allowed_commands
            && !allowed_commands.is_empty()
        {
            let is_allowed = allowed_commands
                .iter()
                .any(|allowed| command.starts_with(allowed));
            if !is_allowed {
                return Err(RuntimeError::Capability {
                    capability: self.metadata.name.clone(),
                    message: format!(
                        "Command not allowed. Allowed commands: {}",
                        allowed_commands.join(", ")
                    ),
                });
            }
            return Ok(());
        }

        if let Some(blocked_pattern) = self
            .config
            .blocked_commands
            .iter()
            .find(|pattern| command.contains(pattern.as_str()))
        {
            return Err(RuntimeError::Capability {
                capability: self.metadata.name.clone(),
                message: format!("Command blocked by policy: {blocked_pattern}"),
            });
        }

        Ok(())
    }
}

impl Default for BashCapability {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CapabilityExecutor for BashCapability {
    async fn execute(&self, args: HashMap<String, Value>) -> CapabilityResult<Value> {
        let command = args
            .get("command")
            .or_else(|| args.get("arg_command"))
            .or_else(|| args.get("arg0"))
            .and_then(|value| value.as_string())
            .ok_or_else(|| RuntimeError::Capability {
                capability: self.metadata.name.clone(),
                message: "Missing required 'command' argument".to_string(),
            })?
            .to_string();

        self.validate_command(&command)?;

        let timeout_secs = args
            .get("timeout")
            .or_else(|| args.get("timeout_secs"))
            .and_then(|value| value.as_u64())
            .unwrap_or(self.config.timeout_secs);

        let mut command_process = Command::new("sh");
        command_process
            .arg("-lc")
            .arg(&command)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Some(working_directory) = &self.config.working_directory {
            command_process.current_dir(working_directory);
        }

        let output =
            tokio::time::timeout(Duration::from_secs(timeout_secs), command_process.output())
                .await
                .map_err(|_| RuntimeError::Timeout {
                    op_id: 0,
                    timeout: Duration::from_secs(timeout_secs),
                })?
                .map_err(|error| RuntimeError::Capability {
                    capability: self.metadata.name.clone(),
                    message: format!("Failed to execute command: {error}"),
                })?;

        let mut payload = String::new();
        payload.push_str(&String::from_utf8_lossy(&output.stdout));
        payload.push_str(&String::from_utf8_lossy(&output.stderr));

        if payload.as_bytes().len() > self.config.max_output_bytes {
            payload = String::from_utf8_lossy(&payload.as_bytes()[..self.config.max_output_bytes])
                .to_string();
            payload.push_str("\n[output truncated]");
        }

        if let Some(code) = output.status.code() {
            payload.push_str(&format!("\n[exit_code={code}]"));
        } else {
            payload.push_str("\n[exit_code=unknown]");
        }

        Ok(Value::String(payload))
    }

    fn metadata(&self) -> &CapabilityMetadata {
        &self.metadata
    }
}
