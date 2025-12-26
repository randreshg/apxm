//! Capability registry for registration and lookup

use super::{executor::CapabilityExecutor, metadata::CapabilityMetadata};
use apxm_core::error::RuntimeError;
use apxm_backends::JsonSchema;
use dashmap::DashMap;
use std::sync::Arc;

/// Result type for registry operations
type RegistryResult<T> = Result<T, RuntimeError>;

/// Concurrent capability registry
///
/// Thread-safe registry for storing and looking up capabilities.
/// Uses DashMap for lock-free concurrent access.
pub struct CapabilityRegistry {
    /// Registered capabilities
    capabilities: Arc<DashMap<String, Arc<dyn CapabilityExecutor>>>,

    /// Compiled JSON schemas for validation
    schemas: Arc<DashMap<String, Arc<JsonSchema>>>,
}

impl CapabilityRegistry {
    /// Create a new empty capability registry
    pub fn new() -> Self {
        Self {
            capabilities: Arc::new(DashMap::new()),
            schemas: Arc::new(DashMap::new()),
        }
    }

    /// Register a capability
    ///
    /// Compiles the capability's JSON schema and stores it for validation.
    ///
    /// # Arguments
    ///
    /// * `capability` - Capability implementation to register
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Capability with same name already exists
    /// - JSON schema is invalid
    pub fn register(&self, capability: Arc<dyn CapabilityExecutor>) -> RegistryResult<()> {
        let metadata = capability.metadata();
        let name = metadata.name.clone();

        // Check for duplicate registration
        if self.capabilities.contains_key(&name) {
            return Err(RuntimeError::Capability {
                capability: name.clone(),
                message: format!("Capability '{}' is already registered", name),
            });
        }

        // Compile JSON schema for validation
        let schema = JsonSchema::from_value(metadata.parameters_schema.clone()).map_err(|e| {
            RuntimeError::Capability {
                capability: name.clone(),
                message: format!("Invalid parameter schema: {}", e),
            }
        })?;

        // Register capability and schema
        self.schemas.insert(name.clone(), Arc::new(schema));
        self.capabilities.insert(name.clone(), capability);

        tracing::info!("Registered capability: {}", name);
        Ok(())
    }

    /// Get a capability by name
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the capability to retrieve
    ///
    /// # Returns
    ///
    /// Arc to the capability executor if found, None otherwise
    pub fn get(&self, name: &str) -> Option<Arc<dyn CapabilityExecutor>> {
        self.capabilities
            .get(name)
            .map(|entry| Arc::clone(entry.value()))
    }

    /// Get the compiled schema for a capability
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the capability
    ///
    /// # Returns
    ///
    /// Reference to the compiled JSON schema if found
    pub fn get_schema(&self, name: &str) -> Option<Arc<JsonSchema>> {
        self.schemas
            .get(name)
            .map(|entry| Arc::clone(entry.value()))
    }

    /// Check if a capability is registered
    pub fn contains(&self, name: &str) -> bool {
        self.capabilities.contains_key(name)
    }

    /// List all registered capability names
    pub fn list_names(&self) -> Vec<String> {
        self.capabilities
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// List all capability metadata
    pub fn list_metadata(&self) -> Vec<CapabilityMetadata> {
        self.capabilities
            .iter()
            .map(|entry| entry.value().metadata().clone())
            .collect()
    }

    /// Unregister a capability
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the capability to remove
    ///
    /// # Returns
    ///
    /// true if capability was removed, false if it wasn't registered
    pub fn unregister(&self, name: &str) -> bool {
        let cap_removed = self.capabilities.remove(name).is_some();
        let schema_removed = self.schemas.remove(name).is_some();
        cap_removed && schema_removed
    }

    /// Get the number of registered capabilities
    pub fn len(&self) -> usize {
        self.capabilities.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.capabilities.is_empty()
    }
}

impl Default for CapabilityRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::executor::EchoCapability;

    #[test]
    fn test_registry_creation() {
        let registry = CapabilityRegistry::new();
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_register_and_get() {
        let registry = CapabilityRegistry::new();
        let echo = Arc::new(EchoCapability::new());

        registry.register(echo).unwrap();

        assert_eq!(registry.len(), 1);
        assert!(registry.contains("echo"));
        assert!(registry.get("echo").is_some());
    }

    #[test]
    fn test_duplicate_registration() {
        let registry = CapabilityRegistry::new();
        let echo1 = Arc::new(EchoCapability::new());
        let echo2 = Arc::new(EchoCapability::new());

        registry.register(echo1).unwrap();
        let result = registry.register(echo2);

        assert!(result.is_err());
    }

    #[test]
    fn test_list_names() {
        let registry = CapabilityRegistry::new();
        let echo = Arc::new(EchoCapability::new());

        registry.register(echo).unwrap();

        let names = registry.list_names();
        assert_eq!(names.len(), 1);
        assert!(names.contains(&"echo".to_string()));
    }

    #[test]
    fn test_unregister() {
        let registry = CapabilityRegistry::new();
        let echo = Arc::new(EchoCapability::new());

        registry.register(echo).unwrap();
        assert_eq!(registry.len(), 1);

        let removed = registry.unregister("echo");
        assert!(removed);
        assert_eq!(registry.len(), 0);
        assert!(!registry.contains("echo"));
    }
}
