//! Multi-backend registry with intelligent routing.
//!
//! Provides backend registration, intelligent routing, health monitoring,
//! and fallback chain execution for robust LLM request handling.

use crate::backends::{LLMBackend, LLMRequest, LLMResponse};
use crate::provider::Provider;
use anyhow::{Context as AnyhowContext, Result};
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

mod health;
mod resolver;

pub use health::{HealthMonitor, HealthStatus};
pub use resolver::{RoutingStrategy, SelectionCriteria};

/// LLM Registry manages multiple backends with routing and fallback.
#[derive(Clone)]
pub struct LLMRegistry {
    /// Registered backends by name
    backends: Arc<DashMap<String, Arc<Provider>>>,
    /// Default backend name
    default_backend: Arc<parking_lot::RwLock<Option<String>>>,
    /// Operation-specific backend defaults
    operation_defaults: Arc<DashMap<String, String>>,
    /// Fallback chains: backend -> list of fallback backends
    fallback_chains: Arc<DashMap<String, Vec<String>>>,
    /// Health monitor
    health_monitor: Arc<HealthMonitor>,
    /// Routing strategy
    routing_strategy: RoutingStrategy,
}

impl LLMRegistry {
    /// Create a new empty registry with default routing.
    pub fn new() -> Self {
        LLMRegistry {
            backends: Arc::new(DashMap::new()),
            default_backend: Arc::new(parking_lot::RwLock::new(None)),
            operation_defaults: Arc::new(DashMap::new()),
            fallback_chains: Arc::new(DashMap::new()),
            health_monitor: Arc::new(HealthMonitor::new()),
            routing_strategy: RoutingStrategy::default(),
        }
    }

    /// Create registry with custom routing strategy.
    pub fn with_strategy(routing_strategy: RoutingStrategy) -> Self {
        LLMRegistry {
            backends: Arc::new(DashMap::new()),
            default_backend: Arc::new(parking_lot::RwLock::new(None)),
            operation_defaults: Arc::new(DashMap::new()),
            fallback_chains: Arc::new(DashMap::new()),
            health_monitor: Arc::new(HealthMonitor::new()),
            routing_strategy,
        }
    }

    /// Register a backend with a given name.
    pub fn register(&self, name: impl Into<String>, backend: Provider) -> Result<()> {
        let name = name.into();
        let backend = Arc::new(backend);

        // Register backend
        self.backends.insert(name.clone(), backend.clone());

        // Initialize health tracking
        self.health_monitor.register_backend(&name);

        Ok(())
    }

    /// Unregister a backend.
    pub fn unregister(&self, name: &str) -> Result<()> {
        self.backends
            .remove(name)
            .with_context(|| format!("Backend '{}' not found", name))?;

        self.health_monitor.unregister_backend(name);

        Ok(())
    }

    /// Set the default backend for requests.
    pub fn set_default(&self, name: impl Into<String>) -> Result<()> {
        let name = name.into();

        // Verify backend exists
        if !self.backends.contains_key(&name) {
            anyhow::bail!("Backend '{}' not registered", name);
        }

        *self.default_backend.write() = Some(name);
        Ok(())
    }

    /// Set operation-specific backend default.
    pub fn set_operation_default(
        &self,
        operation: impl Into<String>,
        backend: impl Into<String>,
    ) -> Result<()> {
        let backend_name = backend.into();

        // Verify backend exists
        if !self.backends.contains_key(&backend_name) {
            anyhow::bail!("Backend '{}' not registered", backend_name);
        }

        self.operation_defaults
            .insert(operation.into(), backend_name);
        Ok(())
    }

    /// Set fallback chain for a backend.
    pub fn set_fallback(&self, backend: impl Into<String>, fallbacks: Vec<String>) -> Result<()> {
        let backend_name = backend.into();

        // Verify all backends exist
        if !self.backends.contains_key(&backend_name) {
            anyhow::bail!("Backend '{}' not registered", backend_name);
        }

        for fallback in &fallbacks {
            if !self.backends.contains_key(fallback) {
                anyhow::bail!("Fallback backend '{}' not registered", fallback);
            }
        }

        self.fallback_chains.insert(backend_name, fallbacks);
        Ok(())
    }

    /// Get registered backend names.
    pub fn backend_names(&self) -> Vec<String> {
        self.backends
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get backend by name.
    pub fn get_backend(&self, name: &str) -> Option<Arc<Provider>> {
        self.backends.get(name).map(|entry| entry.value().clone())
    }

    /// Get health status of a backend.
    pub fn backend_health(&self, name: &str) -> HealthStatus {
        self.health_monitor.status(name)
    }

    /// Generate a response using intelligent routing.
    pub async fn generate(&self, request: LLMRequest) -> Result<LLMResponse> {
        // Resolve which backend to use
        let backend_name = self.resolve_backend(&request)?;

        // Try primary backend
        match self.try_generate(&backend_name, request.clone()).await {
            Ok(response) => Ok(response),
            Err(e) => {
                // Check for fallback chain
                if let Some(fallbacks) = self.fallback_chains.get(&backend_name) {
                    for fallback_name in fallbacks.value() {
                        match self.try_generate(fallback_name, request.clone()).await {
                            Ok(response) => {
                                tracing::info!(
                                    "Fallback successful: {} -> {}",
                                    backend_name,
                                    fallback_name
                                );
                                return Ok(response);
                            }
                            Err(fallback_err) => {
                                tracing::warn!(
                                    "Fallback '{}' failed: {}",
                                    fallback_name,
                                    fallback_err
                                );
                                continue;
                            }
                        }
                    }
                }

                // All attempts exhausted
                Err(e).context(format!(
                    "Request failed on '{}' with no successful fallback",
                    backend_name
                ))
            }
        }
    }

    /// Generate using a specific backend by name.
    pub async fn generate_with_backend(
        &self,
        backend_name: &str,
        request: LLMRequest,
    ) -> Result<LLMResponse> {
        self.try_generate(backend_name, request).await
    }

    /// Try to generate using a specific backend, tracking health and cost.
    async fn try_generate(&self, backend_name: &str, request: LLMRequest) -> Result<LLMResponse> {
        let backend = self
            .backends
            .get(backend_name)
            .with_context(|| format!("Backend '{}' not found", backend_name))?
            .clone();

        // Check health status
        let health = self.health_monitor.status(backend_name);
        if health == HealthStatus::Unhealthy {
            anyhow::bail!("Backend '{}' is unhealthy", backend_name);
        }

        let start = Instant::now();
        let result = backend.generate(request).await;
        let latency = start.elapsed();

        match result {
            Ok(response) => {
                // Record success
                self.health_monitor.record_success(backend_name, latency);
                Ok(response)
            }
            Err(e) => {
                // Record failure
                self.health_monitor.record_failure(backend_name, latency);
                Err(e)
            }
        }
    }

    /// Resolve which backend to use for a request.
    fn resolve_backend(&self, request: &LLMRequest) -> Result<String> {
        // Use resolver to determine backend
        let criteria = SelectionCriteria::from_request(request);
        resolver::resolve(
            &criteria,
            &self.backends,
            &self.operation_defaults,
            &self.default_backend,
            &self.health_monitor,
            &self.routing_strategy,
        )
    }

    /// Perform health checks on all backends.
    pub async fn check_all_backends(&self) -> HashMap<String, HealthStatus> {
        let mut results = HashMap::new();

        for entry in self.backends.iter() {
            let name = entry.key().clone();
            let backend = entry.value();

            let status = match backend.health_check().await {
                Ok(_) => HealthStatus::Healthy,
                Err(_) => HealthStatus::Unhealthy,
            };

            self.health_monitor.set_status(&name, status);
            results.insert(name, status);
        }

        results
    }
}

impl Default for LLMRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::openai::OpenAIBackend;
    use crate::provider::Provider;

    #[tokio::test]
    async fn test_registry_registration() -> Result<(), Box<dyn std::error::Error>> {
        let registry = LLMRegistry::new();

        // Mock backend (would need real API key for actual test)
        let backend = Provider::OpenAI(OpenAIBackend::new("test-key", None).await?);

        registry.register("test", backend)?;

        assert_eq!(registry.backend_names(), vec!["test"]);
        assert!(registry.get_backend("test").is_some());

        Ok(())
    }

    #[test]
    fn test_registry_defaults() {
        let registry = LLMRegistry::new();

        // Can't set default for non-existent backend
        assert!(registry.set_default("nonexistent").is_err());
    }

    #[test]
    fn test_fallback_chain_validation() {
        let registry = LLMRegistry::new();

        // Can't set fallback for non-existent backend
        assert!(
            registry
                .set_fallback("nonexistent", vec!["other".to_string()])
                .is_err()
        );
    }
}
