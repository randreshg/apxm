//! Capability executor trait and execution infrastructure

use super::metadata::CapabilityMetadata;
use apxm_core::{error::RuntimeError, types::values::Value};
use async_trait::async_trait;
use std::collections::HashMap;

/// Result type for capability operations
pub type CapabilityResult<T> = Result<T, RuntimeError>;

/// Trait for capability implementations
///
/// Capabilities are executable tools/functions that can be invoked
/// by the APxM runtime. Each capability must provide metadata and
/// an async execution method.
#[async_trait]
pub trait CapabilityExecutor: Send + Sync {
    /// Execute the capability with given arguments
    ///
    /// # Arguments
    ///
    /// * `args` - Input arguments as key-value pairs
    ///
    /// # Returns
    ///
    /// Result value from capability execution
    ///
    /// # Errors
    ///
    /// Returns RuntimeError::Capability if execution fails
    async fn execute(&self, args: HashMap<String, Value>) -> CapabilityResult<Value>;

    /// Get capability metadata
    ///
    /// Provides schema, description, and other metadata
    /// used for validation and introspection
    fn metadata(&self) -> &CapabilityMetadata;
}

/// Built-in echo capability for testing
pub struct EchoCapability {
    metadata: CapabilityMetadata,
}

impl EchoCapability {
    pub fn new() -> Self {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to echo back"
                }
            },
            "required": ["message"]
        });

        Self {
            metadata: CapabilityMetadata::new("echo", "Echo a message back to the caller", schema)
                .with_returns("string")
                .with_latency(10),
        }
    }
}

impl Default for EchoCapability {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CapabilityExecutor for EchoCapability {
    async fn execute(&self, args: HashMap<String, Value>) -> CapabilityResult<Value> {
        let message = args
            .get("message")
            .and_then(|v| v.as_string())
            .ok_or_else(|| RuntimeError::Capability {
                capability: "echo".to_string(),
                message: "Missing or invalid 'message' argument".to_string(),
            })?;

        Ok(Value::String(format!("Echo: {}", message)))
    }

    fn metadata(&self) -> &CapabilityMetadata {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_echo_capability() -> Result<(), Box<dyn std::error::Error>> {
        let echo = EchoCapability::new();

        let mut args = HashMap::new();
        args.insert("message".to_string(), Value::String("Hello".to_string()));

        let result = echo.execute(args).await?;
        assert_eq!(result.as_string().map(|s| s.as_str()), Some("Echo: Hello"));

        Ok(())
    }

    #[tokio::test]
    async fn test_echo_capability_missing_arg() {
        let echo = EchoCapability::new();
        let args = HashMap::new();

        let result = echo.execute(args).await;
        assert!(result.is_err());
    }
}
