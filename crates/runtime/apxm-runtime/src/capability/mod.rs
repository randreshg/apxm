//! Capability system for tool/function invocation
//!
//! The capability system provides a plugin architecture for external tools and functions.
//! Capabilities can be registered, validated, and invoked through a unified interface.
//!
//! # Architecture
//!
//! - **CapabilityExecutor**: Trait for capability implementations
//! - **CapabilityMetadata**: Schema and metadata for capabilities
//! - **CapabilityRegistry**: Thread-safe storage and lookup
//! - **CapabilitySystem**: Coordinator for invocation with validation
//!
//! # Example
//!
//! ```rust
//! use apxm_runtime::capability::{CapabilitySystem, executor::EchoCapability};
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let system = CapabilitySystem::new();
//!
//! // Register built-in capability
//! system.register(Arc::new(EchoCapability::new()))?;
//!
//! // Invoke capability
//! let mut args = std::collections::HashMap::new();
//! args.insert("message".to_string(), apxm_core::types::values::Value::String("Hello".to_string()));
//! let result = system.invoke("echo", args).await?;
//! # Ok(())
//! # }
//! ```

pub mod executor;
pub mod flow_registry;
pub mod metadata;
pub mod registry;

use crate::aam::{Aam, TransitionLabel};
use apxm_core::{error::RuntimeError, types::values::Value};
use executor::CapabilityExecutor;
use metadata::CapabilityMetadata;
use registry::CapabilityRegistry;
use std::{collections::HashMap, sync::Arc, time::Duration};

/// Result type for capability operations
type CapabilityResult<T> = Result<T, RuntimeError>;

/// Main capability system coordinator
///
/// Provides high-level API for capability invocation with:
/// - Automatic input validation against JSON schemas
/// - Timeout enforcement
/// - Error handling and logging
pub struct CapabilitySystem {
    registry: Arc<CapabilityRegistry>,
    default_timeout: Duration,
    aam: Option<Aam>,
}

impl CapabilitySystem {
    /// Create a new capability system
    pub fn new() -> Self {
        Self {
            registry: Arc::new(CapabilityRegistry::new()),
            default_timeout: Duration::from_secs(30),
            aam: None,
        }
    }

    /// Create with custom default timeout
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            registry: Arc::new(CapabilityRegistry::new()),
            default_timeout: timeout,
            aam: None,
        }
    }

    /// Create with AAM integration (default timeout)
    pub fn with_aam(aam: Aam) -> Self {
        Self {
            registry: Arc::new(CapabilityRegistry::new()),
            default_timeout: Duration::from_secs(30),
            aam: Some(aam),
        }
    }

    /// Get read-only access to the capability registry
    ///
    /// This allows inspection of registered capabilities without
    /// modification access.
    pub fn registry(&self) -> &CapabilityRegistry {
        &self.registry
    }

    /// List all registered capability names
    ///
    /// This is useful for:
    /// - Validating DSL before compilation
    /// - Showing available capabilities to users
    /// - Passing to LLM for constrained generation
    pub fn list_capability_names(&self) -> Vec<String> {
        self.registry.list_names()
    }

    /// Register a capability
    ///
    /// # Arguments
    ///
    /// * `capability` - Capability implementation to register
    ///
    /// # Errors
    ///
    /// Returns error if capability is already registered or has invalid schema
    pub fn register(&self, capability: Arc<dyn CapabilityExecutor>) -> CapabilityResult<()> {
        let metadata = capability.metadata().clone();
        self.registry.register(Arc::clone(&capability))?;

        if let Some(aam) = &self.aam {
            let label = TransitionLabel::custom(format!("register_capability:{}", metadata.name));
            aam.register_capability(metadata.name.clone(), (&metadata).into(), label);
        }

        Ok(())
    }

    /// Invoke a capability by name with validation
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the capability to invoke
    /// * `args` - Input arguments as key-value pairs
    ///
    /// # Returns
    ///
    /// Result value from capability execution
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Capability not found
    /// - Input validation fails
    /// - Execution times out
    /// - Capability execution fails
    pub async fn invoke(
        &self,
        name: &str,
        args: HashMap<String, Value>,
    ) -> CapabilityResult<Value> {
        self.invoke_with_timeout(name, args, self.default_timeout)
            .await
    }

    /// Invoke capability with custom timeout
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the capability to invoke
    /// * `args` - Input arguments
    /// * `timeout` - Maximum execution time
    pub async fn invoke_with_timeout(
        &self,
        name: &str,
        args: HashMap<String, Value>,
        timeout: Duration,
    ) -> CapabilityResult<Value> {
        // Get capability
        let capability = self
            .registry
            .get(name)
            .ok_or_else(|| RuntimeError::Capability {
                capability: name.to_string(),
                message: format!(
                    "Capability '{}' not found. Available: {}",
                    name,
                    self.registry.list_names().join(", ")
                ),
            })?;

        // Validate arguments against schema
        self.validate_args(name, &args).await?;

        tracing::debug!(capability = %name, "Invoking capability");

        // Execute with timeout
        let result = tokio::time::timeout(timeout, capability.execute(args))
            .await
            .map_err(|_| RuntimeError::Timeout { op_id: 0, timeout })?
            .map_err(|e| {
                tracing::error!(capability = %name, error = %e, "Capability execution failed");
                e
            })?;

        tracing::info!(capability = %name, "Capability executed successfully");
        Ok(result)
    }

    /// Validate arguments against capability schema
    async fn validate_args(
        &self,
        name: &str,
        args: &HashMap<String, Value>,
    ) -> CapabilityResult<()> {
        let schema = self
            .registry
            .get_schema(name)
            .ok_or_else(|| RuntimeError::Capability {
                capability: name.to_string(),
                message: "Schema not found".to_string(),
            })?;

        // Convert args to JSON value
        let args_json = serde_json::to_value(args).map_err(|e| RuntimeError::Capability {
            capability: name.to_string(),
            message: format!("Failed to serialize arguments: {}", e),
        })?;

        // Validate against schema
        schema
            .validate(&args_json)
            .map_err(|e| RuntimeError::Capability {
                capability: name.to_string(),
                message: format!("Input validation failed: {}", e),
            })?;

        Ok(())
    }

    /// List all registered capabilities
    pub fn list_capabilities(&self) -> Vec<CapabilityMetadata> {
        self.registry.list_metadata()
    }

    /// Check if capability exists
    pub fn has_capability(&self, name: &str) -> bool {
        self.registry.contains(name)
    }

    /// Get capability metadata
    pub fn get_metadata(&self, name: &str) -> Option<CapabilityMetadata> {
        self.registry.get(name).map(|cap| cap.metadata().clone())
    }
}

impl Default for CapabilitySystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aam::Aam;
    use executor::EchoCapability;

    #[tokio::test]
    async fn test_capability_system_creation() {
        let system = CapabilitySystem::new();
        assert_eq!(system.list_capabilities().len(), 0);
    }

    #[tokio::test]
    async fn test_register_and_invoke() {
        let system = CapabilitySystem::new();
        assert!(system.register(Arc::new(EchoCapability::new())).is_ok());

        let mut args = HashMap::new();
        args.insert("message".to_string(), Value::String("Test".to_string()));

        let result = system.invoke("echo", args).await;
        match result {
            Ok(val) => assert_eq!(val.as_string().map(|s| s.as_str()), Some("Echo: Test")),
            Err(e) => panic!("invoke failed: {}", e),
        }
    }

    #[tokio::test]
    async fn test_invoke_nonexistent_capability() {
        let system = CapabilitySystem::new();

        let result = system.invoke("nonexistent", HashMap::new()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_validation_failure() {
        let system = CapabilitySystem::new();
        assert!(system.register(Arc::new(EchoCapability::new())).is_ok());

        // Missing required argument
        let result = system.invoke("echo", HashMap::new()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_has_capability() {
        let system = CapabilitySystem::new();
        assert!(system.register(Arc::new(EchoCapability::new())).is_ok());

        assert!(system.has_capability("echo"));
        assert!(!system.has_capability("nonexistent"));
    }

    #[tokio::test]
    async fn test_get_metadata() {
        let system = CapabilitySystem::new();
        assert!(system.register(Arc::new(EchoCapability::new())).is_ok());

        let metadata = match system.get_metadata("echo") {
            Some(m) => m,
            None => panic!("metadata missing for 'echo'"),
        };
        assert_eq!(metadata.name, "echo");
        assert!(metadata.description.to_lowercase().contains("echo"));
    }

    #[tokio::test]
    async fn test_register_records_in_aam() {
        let aam = Aam::new();
        let system = CapabilitySystem::with_aam(aam.clone());
        assert!(system.register(Arc::new(EchoCapability::new())).is_ok());
        let capabilities = aam.capabilities();
        assert!(capabilities.contains_key("echo"));
    }
}
