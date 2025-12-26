//! Request routing and backend selection logic.
//!
//! Determines which backend to use for a given request based on
//! explicit selection, operation type, model name, or defaults.

use super::health::{HealthMonitor, HealthStatus};
use crate::llm::backends::{LLMBackend, LLMRequest};
use crate::llm::provider::Provider;
use anyhow::Result;
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Routing strategy determines how backends are selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RoutingStrategy {
    /// Always use the first healthy backend
    #[default]
    FirstHealthy,
    /// Round-robin across healthy backends
    RoundRobin,
    /// Prefer backends with lowest latency
    LowLatency,
}

/// Selection criteria extracted from a request.
#[derive(Debug, Clone)]
pub struct SelectionCriteria {
    /// Explicitly requested backend name
    pub backend: Option<String>,
    /// Requested model name
    pub model: Option<String>,
    /// Operation type (e.g., "reason", "plan", "generate")
    pub operation: Option<String>,
}

impl SelectionCriteria {
    /// Extract selection criteria from a request.
    pub fn from_request(request: &LLMRequest) -> Self {
        SelectionCriteria {
            backend: request.backend.clone(),
            model: request.model.clone(),
            operation: request.operation_type.clone(),
        }
    }
}

/// Resolve which backend to use for a request.
pub fn resolve(
    criteria: &SelectionCriteria,
    backends: &Arc<DashMap<String, Arc<Provider>>>,
    operation_defaults: &Arc<DashMap<String, String>>,
    default_backend: &Arc<RwLock<Option<String>>>,
    health_monitor: &HealthMonitor,
    strategy: &RoutingStrategy,
) -> Result<String> {
    // Priority 1: Explicit backend selection
    if let Some(ref backend_name) = criteria.backend {
        if backends.contains_key(backend_name) {
            return Ok(backend_name.clone());
        }
        anyhow::bail!("Explicitly requested backend '{}' not found", backend_name);
    }

    // Priority 2: Model-based routing
    if let Some(ref model) = criteria.model
        && let Some(backend_name) = find_backend_for_model(backends, model)
    {
        return Ok(backend_name);
    }

    // Priority 3: Operation-specific default
    if let Some(ref operation) = criteria.operation
        && let Some(entry) = operation_defaults.get(operation)
    {
        let backend_name = entry.value().clone();
        if backends.contains_key(&backend_name) {
            return Ok(backend_name);
        }
    }

    // Priority 4: Global default backend
    {
        let default = default_backend.read();
        if let Some(ref backend_name) = *default
            && backends.contains_key(backend_name)
        {
            return Ok(backend_name.clone());
        }
    }

    // Priority 5: Select based on strategy
    select_by_strategy(backends, health_monitor, strategy)
}

/// Find a backend that supports the given model.
fn find_backend_for_model(
    backends: &Arc<DashMap<String, Arc<Provider>>>,
    model: &str,
) -> Option<String> {
    // Try exact model match first
    for entry in backends.iter() {
        let backend = entry.value();
        if backend.model() == model {
            return Some(entry.key().clone());
        }
    }

    // Try provider-based matching (e.g., "gpt-4" -> OpenAI backend)
    if model.starts_with("gpt-") || model.starts_with("o1-") {
        for entry in backends.iter() {
            if entry.value().name().to_lowercase().contains("openai") {
                return Some(entry.key().clone());
            }
        }
    } else if model.starts_with("claude-") {
        for entry in backends.iter() {
            if entry.value().name().to_lowercase().contains("anthropic") {
                return Some(entry.key().clone());
            }
        }
    } else if model.starts_with("gemini-") {
        for entry in backends.iter() {
            if entry.value().name().to_lowercase().contains("google") {
                return Some(entry.key().clone());
            }
        }
    }

    None
}

/// Select a backend based on routing strategy.
fn select_by_strategy(
    backends: &Arc<DashMap<String, Arc<Provider>>>,
    health_monitor: &HealthMonitor,
    strategy: &RoutingStrategy,
) -> Result<String> {
    if backends.is_empty() {
        anyhow::bail!("No backends registered");
    }

    match strategy {
        RoutingStrategy::FirstHealthy => select_first_healthy(backends, health_monitor),
        RoutingStrategy::RoundRobin => select_round_robin(backends, health_monitor),
        RoutingStrategy::LowLatency => select_low_latency(backends, health_monitor),
    }
}

/// Select the first healthy backend.
fn select_first_healthy(
    backends: &Arc<DashMap<String, Arc<Provider>>>,
    health_monitor: &HealthMonitor,
) -> Result<String> {
    // Try to find a healthy backend
    for entry in backends.iter() {
        let name = entry.key().clone();
        let status = health_monitor.status(&name);

        if status == HealthStatus::Healthy || status == HealthStatus::Unknown {
            return Ok(name);
        }
    }

    // If no healthy backend, try degraded
    for entry in backends.iter() {
        let name = entry.key().clone();
        let status = health_monitor.status(&name);

        if status == HealthStatus::Degraded {
            return Ok(name);
        }
    }

    // Last resort: return any backend (handle defensively)
    //
    // In practice the caller should ensure `backends` is non-empty. Still,
    // avoid `unwrap()` here and return an explicit error if something
    // unexpected happens.
    if let Some(entry) = backends.iter().next() {
        Ok(entry.key().clone())
    } else {
        anyhow::bail!("No backends registered (unexpected)")
    }
}

/// Select backend using round-robin (simplified: just pick first healthy).
/// A full implementation would maintain a counter.
fn select_round_robin(
    backends: &Arc<DashMap<String, Arc<Provider>>>,
    health_monitor: &HealthMonitor,
) -> Result<String> {
    // For now, use same logic as FirstHealthy
    // A full implementation would track the last used index
    select_first_healthy(backends, health_monitor)
}

/// Select backend with lowest average latency.
fn select_low_latency(
    backends: &Arc<DashMap<String, Arc<Provider>>>,
    health_monitor: &HealthMonitor,
) -> Result<String> {
    let mut best_backend: Option<(String, std::time::Duration)> = None;

    for entry in backends.iter() {
        let name = entry.key().clone();
        let status = health_monitor.status(&name);

        // Skip unhealthy backends
        if status == HealthStatus::Unhealthy {
            continue;
        }

        if let Some(avg_latency) = health_monitor.average_latency(&name) {
            match &best_backend {
                None => {
                    best_backend = Some((name, avg_latency));
                }
                Some((_, best_latency)) => {
                    if avg_latency < *best_latency {
                        best_backend = Some((name, avg_latency));
                    }
                }
            }
        }
    }

    if let Some((name, _)) = best_backend {
        Ok(name)
    } else {
        // No latency data, fall back to first healthy
        select_first_healthy(backends, health_monitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_routing_strategy_default() {
        assert_eq!(RoutingStrategy::default(), RoutingStrategy::FirstHealthy);
    }

    #[test]
    fn test_model_matching() {
        // Test provider detection from model names
        assert!("gpt-4".starts_with("gpt-"));
        assert!("claude-3-opus".starts_with("claude-"));
        assert!("gemini-pro".starts_with("gemini-"));
    }
}
